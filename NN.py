def NN(random_seed,
       samples_count,
       frequencies_count,
       dimension,
       topology,
       activations,
       batchsize,
       resample_factor,
       lr,
       passes,
       loss,
       loss_lambda_parameter,
       characteristic_function,
       params,
       printLoss,
       makeplots):

    #---------------------------------------- Torch setup
    import torch
    print("############################################")
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    print("############################################")
    
    #---------------------------------------- Random seed setup
    if not random_seed:
        torch.manual_seed(1)
    
    #---------------------------------------- Import time module
    import time

    def define_topology(dimension, topology):
        topology = [dimension] + topology
        weights = []
        biases = []
        for i in range(len(topology) - 1):
            weights.append(2 * torch.rand(topology[i + 1], topology[i], device = device) - 1)
            biases.append(torch.zeros(topology[i + 1], 1, device = device))
            
        return{
            "weights": weights,
            "biases": biases
        }

    def layer(samples, weights, biases, activation):
        # Broadcast biases (automatic)
        # Linear pre-activation function
        Z = weights@samples.T + biases
        
        # Activation functions
        activations = {
            "sigmoid": lambda x: 1 / (1 + torch.exp(-x)),
            "ReLU": lambda x: torch.clamp(x, min = 0),
            "purelin": lambda x: x,
            "sin": torch.sin,
            "tanh": torch.tanh
        }
        
        # Derivative functions
        derivatives = {
            "sigmoid": lambda x: (1 / (1 + torch.exp(-x))) * (1 - (1 / (1 + torch.exp(-x)))),
            "ReLU":    lambda x: (x > 0).float(), # Returns a bool, hence convert to float
            "purelin": lambda x: torch.ones_like(x),
            "sin":     torch.cos,
            "tanh":    lambda x: 1 - torch.tanh(x)**2
        }
        
        activation_function = activations[activation]
        derivative_function = derivatives[activation]
        
        # Post-activation function
        Y = activation_function(Z).T
        
        return{
            "Y": Y,
            "pre_activation": Z,
            "derivative_function": derivative_function
        }

    def forwardpass(samples, weights, biases, activations):
        layers = []
        input = samples
        for i in range(len(weights)):
            layers.append(layer(input, weights[i], biases[i], activations[i]))
            input = layers[i]["Y"]
            
        return layers
        
    def calculateCF(x, t):
        n = x.shape[0]
        TX_tr = t @ x.T
        col_of_ones = torch.ones(n, 1, device = device)
        empCF_real = (torch.cos(TX_tr) @ col_of_ones) / n
        empCF_imaginary = (torch.sin(TX_tr) @ col_of_ones) / n
        empCF = torch.cat([empCF_real, empCF_imaginary], dim = 1)
        
        return{
            "empCF": empCF,
            "TX_tr": TX_tr
        }

    def execute(samples_count, frequencies_count, dimension, weights, biases, activations, resample_factor, loss, loss_lambda_parameter, lr, passes, batchsize, printLoss):
        
        startTime = time.time() # Begin timer
        counter = 0 # Begin pass counter
        loss_vector = []
        
        # Initialise RMSProp
        vW = [torch.zeros_like(W) for W in weights]
        vb = [torch.zeros_like(b) for b in biases]
        beta = 0.9
        eps = 1e-8
        
        for i in range(passes):
            
            if i % resample_factor == 0:
                samples = torch.randn(samples_count, dimension, device = device) # Initialise new samples
                frequencies = torch.randn(frequencies_count, dimension, device = device) # initialise new frequencies
                targets = compute_targets(characteristic_function, frequencies, params) # Initialise new targets
            
            # Creating batch
            if batchsize == "fullbatch":
                batchsize = samples_count
            else:
                batchsize = batchsize
            batchsamples = samples[torch.randperm(samples.shape[0])[:batchsize], :]
            batch_n = batchsamples.shape[0]
            
            # Forward pass
            layers = forwardpass(batchsamples, weights, biases, activations)
            X = layers[-1]["Y"]
            
            # CF calculation
            empCFstuff = calculateCF(X, frequencies) # I made this function because I need to calculate the empCF again later
            empCF = empCFstuff["empCF"]
            TX_tr = empCFstuff["TX_tr"]
            
            # First derivative term
            if loss == "SEL":
                dL_dempCF = 2 * (empCF - targets)
                dL_real = dL_dempCF[:, 0].unsqueeze(1) # To keep it a tensor of m x 1
                dL_imaginary = dL_dempCF[:, 1].unsqueeze(1) # To keep it a tensor of m x 1
            elif loss == "SEL+":
                dL_dempCF = 2 * (empCF - targets) + 4 * loss_lambda_parameter * (empCF - targets) @ ((empCF - targets).T @ (empCF - targets) - torch.diag(torch.diag((empCF - targets).T @ (empCF - targets))))
                dL_real = dL_dempCF[:, 0].unsqueeze(1) # To keep it a tensor of m x 1
                dL_imaginary = dL_dempCF[:, 1].unsqueeze(1) # To keep it a tensor of m x 1
                
            # Second derivative term
            sins = torch.sin(TX_tr)
            coses = torch.cos(TX_tr)
            
            dL_dX = (frequencies.T @ (-sins * dL_real + coses * dL_imaginary)) / batch_n
            dL_dX = dL_dX.T
            
            # Third derivative term
            dX_dZ = layers[-1]["derivative_function"](layers[-1]["pre_activation"].T)
            dL_dZ = dL_dX * dX_dZ
            
            # Back propagation
            delta = [None] * len(weights)
            
            for j in reversed(range(len(weights))):
                if j != (len(weights) - 1):
                    delta[j] = (delta[j + 1] @ weights[j + 1]) * layers[j]["derivative_function"](layers[j]["pre_activation"].T)
                else:
                    delta[j] = dL_dZ
            for j in range(len(weights)): # Using RMSProp
                if j == 0:
                    #weights[j] = weights[j] - lr * delta[j].T @ batchsamples
                    vW[j] = beta * vW[j] + (1 - beta) * (delta[j].T @ batchsamples)**2
                    weights[j] = weights[j] - lr * (delta[j].T @ batchsamples) / (torch.sqrt(vW[j]) + eps)
                else:
                    #weights[j] = weights[j] - lr * delta[j].T @ layers[j - 1]["Y"]
                    vW[j] = beta * vW[j] + (1 - beta) * (delta[j].T @ layers[j - 1]["Y"])**2
                    weights[j] = weights[j] - lr * (delta[j].T @ layers[j - 1]["Y"]) / (torch.sqrt(vW[j]) + eps)
                
                #biases[j] = biases[j] - lr * delta[j].T @ torch.ones((delta[j].shape[0], 1), device = device)
                vb[j] = beta * vb[j] + (1 - beta) * (delta[j].T @ torch.ones((delta[j].shape[0], 1), device = device))**2
                biases[j] = biases[j] - lr * (delta[j].T @ torch.ones((delta[j].shape[0], 1), device = device)) / (torch.sqrt(vb[j]) + eps)
            
            counter = counter + 1
            
            if loss == "SEL":
                currentloss = torch.trace((targets - empCF).T @ (targets - empCF)).item() # item converts to python float
            elif loss == "SEL+":
                A = (targets - empCF).T @ (targets - empCF)
                squared_error_term = torch.trace(A)
                diag_matrix = torch.diag(torch.diag(A))
                argument = A - diag_matrix
                squared_error_of_off_diagonal = torch.trace(argument.T @ argument)
                currentloss = (squared_error_term + loss_lambda_parameter * squared_error_of_off_diagonal).item() # item converts to python float
            loss_vector.append(currentloss)
            if printLoss and (counter % 100 == 0):
                print("Pass:", counter, "Loss:", currentloss)
                
        endTime = time.time() # End timer
        timeTook = endTime - startTime
        print("Training took", round(timeTook, 2), "seconds.")
        
        return{
            "Y_lastlayer": layers[-1]["Y"],
            "weights": weights,
            "biases": biases,
            "lossvals": loss_vector,
            "training_time": timeTook
        }
        
    def compute_targets(characteristic_function, frequencies, params):
        params = [torch.tensor(par, dtype = torch.complex64, device = device) for par in params] # dtype = complex to avoid float v long v complex issues
        def cf_mvnormal(t):
            t = t.to(torch.complex64)
            mu, var = params
            result = []
            for i in range(t.shape[0]):
                result.append(torch.exp(1j * mu.T @ t[i, :] - 0.5 * t[i, :].T @ var @ t[i, :]))
            result = torch.stack(result) # Turns python list into torch vector
            return torch.cat([result.real.unsqueeze(1), result.imag.unsqueeze(1)], dim = 1)
        
        def cf_mvnormal_mix(t):
            t = t.to(torch.complex64)
            mu1, var1, mu2, var2, mixture_weight = params
            result = []
            for i in range(t.shape[0]):
                mv1 = torch.exp(1j * mu1.T @ t[i, :] - 0.5 * t[i, :].T @ var1 @ t[i, :])
                mv2 = torch.exp(1j * mu2.T @ t[i, :] - 0.5 * t[i, :].T @ var2 @ t[i, :])
                result.append(mixture_weight * mv1 + (1 - mixture_weight) * mv2)
            result = torch.stack(result)
            return torch.cat([result.real.unsqueeze(1), result.imag.unsqueeze(1)], dim = 1)
        
        def cf_eq_MNIG_mixture(t):
            t = t.to(torch.complex64)
            m1, b1, var1, a1, d1, m2, b2, var2, a2, d2 = params
            
            g1 = torch.sqrt(a1 ** 2 - b1.T @ var1 @ b1)
            g2 = torch.sqrt(a2 ** 2 - b2.T @ var2 @ b2)
            
            result = []
            for i in range(t.shape[0]):
                MNIG1 = torch.exp(1j * t[i, :].T @ m1 + d1 * (g1 - torch.sqrt(a1 ** 2 - (b1 + 1j * t[i, :]).T @ var1 @ (b1 + 1j * t[i, :]))))
                MNIG2 = torch.exp(1j * t[i, :].T @ m2 + d2 * (g2 - torch.sqrt(a2 ** 2 - (b2 + 1j * t[i, :]).T @ var2 @ (b2 + 1j * t[i, :]))))
                result.append(0.5 * MNIG1 + 0.5 * MNIG2)
            result = torch.stack(result) # Turns python list into torch vector
            return torch.cat([result.real.unsqueeze(1), result.imag.unsqueeze(1)], dim = 1)
        
        CF_function = {
            "eq_MNIG_mixture": cf_eq_MNIG_mixture,
            "mvnormal": cf_mvnormal,
            "mvnormal_mix": cf_mvnormal_mix
        }
        
        targets = CF_function[characteristic_function](frequencies)
        
        return targets
    
    def compute_density(characteristic_function, grid, params):
        params = [torch.tensor(par, dtype = torch.float64, device = device) for par in params] # The dtype ensure the cals dont complain about long v float stuff
        def density_mvnormal(x):
            mu = params[0]
            var = params[1]
            var_inv = torch.linalg.inv(var)
            det_var = torch.linalg.det(var)
            constant = 1 / (2 * torch.pi * torch.sqrt(det_var))
            result = []
            for i in range(x.shape[0]):
                result.append(constant * torch.exp(-0.5 * (x[i, :] - mu).T @ var_inv @ (x[i, :] - mu)))
            result = torch.stack(result) # Turns python list into torch vector
            return result
        
        def density_mvnormal_mix(x):
            mu1, var1, mu2, var2, mixture_weight = params
            var1_inv = torch.linalg.inv(var1)
            det_var1 = torch.linalg.det(var1)
            constant1 = 1 / (2 * torch.pi * torch.sqrt(det_var1))
            var2_inv = torch.linalg.inv(var2)
            det_var2 = torch.linalg.det(var2)
            constant2 = 1 / (2 * torch.pi * torch.sqrt(det_var2))
            result = []
            for i in range(x.shape[0]):
                mv1 = constant1 * torch.exp(-0.5 * (x[i, :] - mu1).T @ var1_inv @ (x[i, :] - mu1))
                mv2 = constant2 * torch.exp(-0.5 * (x[i, :] - mu2).T @ var2_inv @ (x[i, :] - mu2))
                result.append(mixture_weight * mv1 + (1 - mixture_weight) * mv2)
            result = torch.stack(result) # Turns python list into torch vector
            return result
        
        def density_eq_MNIG_mixture(x):
            m1, b1, var1, a1, d1, m2, b2, var2, a2, d2 = params
            p = m1.shape[0]
            inv_var1 = torch.linalg.inv(var1)
            inv_var2 = torch.linalg.inv(var2)
            det_var1 = torch.linalg.det(var1)
            det_var2 = torch.linalg.det(var2)
            g1 = torch.sqrt(a1 ** 2 - b1.T @ var1 @ b1)
            g2 = torch.sqrt(a2 ** 2 - b2.T @ var2 @ b2)
            
            # Hard coding the case where p = 2 for the Bessel function
            def K(y):
                constant = torch.tensor(torch.pi / 2, device = device) # Converting the float into a tensor so torch.sqrt works
                return torch.sqrt(constant) * torch.exp(-y) * y ** (-3 / 2) * (1 + y)
            
            result = []
            for i in range(x.shape[0]):
                P1 = d1 * g1 + b1.T @ (x[i, :] - m1)
                P2 = d2 * g2 + b2.T @ (x[i, :] - m2)
                Q1 = torch.sqrt(d1 ** 2 + ((x[i, :] - m1).T @ inv_var1 @ (x[i, :] - m1)))
                Q2 = torch.sqrt(d2 ** 2 + ((x[i, :] - m2).T @ inv_var2 @ (x[i, :] - m2)))
                firsthalf1 = (d1 / 2 ** ((p - 1) / 2)) * (a1 / (torch.pi * Q1)) ** ((p + 1) / 2)
                firsthalf2 = (d2 / 2 ** ((p - 1) / 2)) * (a2 / (torch.pi * Q2)) ** ((p + 1) / 2)
                secondhalf1 = torch.exp(P1) * K(a1 * Q1)
                secondhalf2 = torch.exp(P2) * K(a2 * Q2)
                MNIG1 = firsthalf1 * secondhalf1
                MNIG2 = firsthalf2 * secondhalf2
                result.append(0.5 * MNIG1 + 0.5 * MNIG2)
            result = torch.stack(result) # Turns python list into torch vector
            return result
        
        Density_function = {
            "mvnormal": density_mvnormal,
            "mvnormal_mix": density_mvnormal_mix,
            "eq_MNIG_mixture": density_eq_MNIG_mixture
        }
        
        density = Density_function[characteristic_function](grid)
        
        return density
        
    def produce_samples(characteristic_function, params, N):
        params = [torch.tensor(par, dtype = torch.float64, device = device) for par in params]
        def mvnormal_samples():
            mean = params[0]
            var = params[1]
            mvnorm = torch.distributions.MultivariateNormal(mean, var)
            the_samples = mvnorm.sample((N, ))
            return the_samples
        
        def mvnormal_mix_samples():
            mu1, var1, mu2, var2, mixture_weight = params
            mv1 = torch.distributions.MultivariateNormal(mu1, var1)
            mv2 = torch.distributions.MultivariateNormal(mu2, var2)
            the_samples1 = mv1.sample((N, ))
            the_samples2 = mv2.sample((N, ))
            indicator = torch.bernoulli(torch.full((N, ), mixture_weight, device = device)).bool()
            the_samples = torch.where(indicator.unsqueeze(1), the_samples1, the_samples2)
            return the_samples
        
        def eq_MNIG_mixture_samples():
            m1, b1, var1, a1, d1, m2, b2, var2, a2, d2 = params
            def sample_inverse_gaussian(mu, l, size):
                v = torch.randn(size, device = device) ** 2
                x = mu + (mu ** 2 * v) / (2 * l) - (mu / (2 * l)) * torch.sqrt(4 * mu * l * v + mu ** 2 * v ** 2)
                u = torch.rand(size, device = device)
                sample = torch.where(u <= mu / (mu + x), x, mu ** 2 / x)
                return sample
                
            g1 = torch.sqrt(a1 ** 2 - b1.T @ var1 @ b1)
            g2 = torch.sqrt(a2 ** 2 - b2.T @ var2 @ b2)
            # Sample V ~ IG
            V1 = sample_inverse_gaussian(d1 / g1, d1 ** 2, N)
            # Sample X | V
            L1 = torch.linalg.cholesky(var1)
            z1 = torch.randn(N, 2, dtype = torch.float64, device = device)  # standard normals
            X1 = m1 + V1[:, None] * b1 + (z1 @ L1.T) * V1[:, None].sqrt()
            # Sample V ~ IG
            V2 = sample_inverse_gaussian(d2 / g2, d2 ** 2, N)
            # Sample X | V
            L2 = torch.linalg.cholesky(var2)
            z2 = torch.randn(N, 2, dtype = torch.float64, device = device)  # standard normals
            X2 = m2 + V2[:, None] * b2 + (z2 @ L2.T) * V2[:, None].sqrt()
            
            mask = torch.rand(N, device = device) < 0.5
            the_samples = torch.zeros_like(X1)
            the_samples[mask] = X1[mask]
            the_samples[~mask] = X2[~mask]
            
            return the_samples
        
        Sampling_function = {
            "mvnormal": mvnormal_samples,
            "mvnormal_mix": mvnormal_mix_samples,
            "eq_MNIG_mixture": eq_MNIG_mixture_samples
        }
        
        the_samples = Sampling_function[characteristic_function]()
        
        return the_samples
    
    def report_statistics(characteristic_function, params, X):
        n, p = X.shape
        mean = X.mean(dim = 0)
        var = torch.cov(X.T) # Transposed because torch.cov does features by samples, not samples by features
        # inv_var = torch.linalg.inv(var)
        # inner_prod = (X - mean) @ inv_var @ (X - mean).T
        # skewness = (inner_prod ** 3).sum() / (n ** 2)
        # kurtosis = (torch.diag(inner_prod) ** 2).mean()
        
        params = [torch.tensor(par, dtype = torch.float64, device = device) for par in params] # The dtype ensure the cals dont complain about long v float stuff
        
        #-------------------- Add future densities in this block
        def stats_mvnormal():
            true_mean = params[0]
            true_var = params[1]
            
            return{
                "true_mean": true_mean,
                "true_var": true_var
            }
            
        def stats_mvnormal_mix():
            mu1, var1, mu2, var2, mixture_weight = params
            true_mean = mixture_weight * mu1 + (1 - mixture_weight) * mu2
            true_var = mixture_weight * var1 + (1 - mixture_weight) * var2 + mixture_weight * (1 - mixture_weight) * (mu1 - mu2) @ (mu1 - mu2).T
            
            return{
                "true_mean": true_mean,
                "true_var": true_var
            }
            
        def stats_eq_MNIG_mixture():
            m1, b1, var1, a1, d1, m2, b2, var2, a2, d2 = params
            g1 = torch.sqrt(a1 ** 2 - b1.T @ var1 @ b1)
            g2 = torch.sqrt(a2 ** 2 - b2.T @ var2 @ b2)
            
            marginal_mean1 = m1 + d1 * (b1 / g1)
            marginal_mean2 = m2 + d2 * (b2 / g2)
            
            true_mean = 0.5 * marginal_mean1 + 0.5 * marginal_mean2
            
            marginal_var1 = d1 * ((var1 / g1) + ((b1 @ b1.T) / g1 ** 3))
            marginal_var2 = d2 * ((var2 / g2) + ((b2 @ b2.T) / g2 ** 3))
            
            true_var = (0.5 * marginal_var1 + 0.5 * marginal_var2) + 0.5 * (marginal_var1 - true_mean) @ (marginal_var1 - true_mean).T + 0.5 * (marginal_var2 - true_mean) @ (marginal_var2 - true_mean).T
            
            return{
                "true_mean": true_mean,
                "true_var": true_var
            }
        
        Stats_distribution = {
            "mvnormal": stats_mvnormal,
            "mvnormal_mix": stats_mvnormal_mix,
            "eq_MNIG_mixture": stats_eq_MNIG_mixture
        }
        #-------------------- Add future densities in this block
        
        stats = Stats_distribution[characteristic_function]()
        
        print("Reconstructed mean is", mean, "\nReconstructed variance is", var, "\n")
        print("True mean is", stats["true_mean"], "\nTrue variance is", stats["true_var"])
    
    def createplots(bool):
        N = 10000
        if bool:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import ListedColormap
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.lines as mlines
            import seaborn as sns
            
            #---------------------------------------- CF plots
            grid_size = 100
            
            x = torch.linspace(-5, 5, grid_size, device = device)
            y = torch.linspace(-5, 5, grid_size, device = device)
            X, Y = torch.meshgrid(x, y, indexing = "ij")
            frequency_grid = torch.stack([X.flatten(), Y.flatten()], dim = 1) # Feed this into compute_targets; calculateCF instead of random frequencies to preserve order
            
            true_CF = compute_targets(characteristic_function, frequency_grid, params) # This is ordered for plotting purposes
            reconstructed_CF = calculateCF(result["Y_lastlayer"], frequency_grid)["empCF"] # This is now also ordered instead of random
            
            Z_true_real = true_CF[:, 0].reshape(grid_size, grid_size).cpu()
            Z_reconstructed_real  = reconstructed_CF[:, 0].reshape(grid_size, grid_size).cpu()
            Z_true_imaginary = true_CF[:, 1].reshape(grid_size, grid_size).cpu()
            Z_reconstructed_imaginary  = reconstructed_CF[:, 1].reshape(grid_size, grid_size).cpu()
            X, Y = X.cpu(), Y.cpu()
            
            fig = plt.figure(figsize = (12, 12))

            ax1 = fig.add_subplot(2, 2, 1, projection = "3d")
            ax1.plot_surface(X, Y, Z_true_real, cmap = "viridis")
            ax1.set_title("Re{True CF}")
            ax1.set_xlabel("t1")
            ax1.set_ylabel("t2")

            ax2 = fig.add_subplot(2, 2, 2, projection = "3d")
            ax2.plot_surface(X, Y, Z_reconstructed_real, cmap = "viridis")
            ax2.set_title("Re{Reconstructed CF}")
            ax2.set_xlabel("t1")
            ax2.set_ylabel("t2")

            ax3 = fig.add_subplot(2, 2, 3, projection = "3d")
            ax3.plot_surface(X, Y, Z_true_imaginary, cmap = "plasma")
            ax3.set_title("Im{True CF}")
            ax3.set_xlabel("t1")
            ax3.set_ylabel("t2")

            ax4 = fig.add_subplot(2, 2, 4, projection = "3d")
            ax4.plot_surface(X, Y, Z_reconstructed_imaginary, cmap = "plasma")
            ax4.set_title("Im{Reconstructed CF}")
            ax4.set_xlabel("t1")
            ax4.set_ylabel("t2")

            plt.tight_layout()
            plt.savefig("CF_plots.png")
            print("Plot saved as CF_plots.png")
            
            #---------------------------------------- Density plot
            # Generating new inputs and pushing them through the finished network
            new_samples = torch.randn(N, dimension, device = device)
            Z_reconstructed = forwardpass(new_samples, result["weights"], result["biases"], activations)[-1]["Y"]
            
            Z_true = produce_samples(characteristic_function, params, N)
            
            Z_true, Z_reconstructed = Z_true.cpu(), Z_reconstructed.cpu()
            
            plt.figure(figsize = (8, 8))
            
            plt.grid(True)
            
            color_cmap = ListedColormap(["black"])
            
            sns.kdeplot(x = Z_reconstructed[:, 0], y = Z_reconstructed[:, 1], fill = True, cmap = "coolwarm", levels = 10)
            sns.kdeplot(x = Z_true[:, 0], y = Z_true[:, 1], fill = False, cmap = color_cmap, levels = 10)
            plt.axhline(0, color = "red", label = "x2 = 0", linestyle = "--", linewidth = 1)
            plt.axvline(0, color = "red", label = "x1 = 0", linestyle = "--", linewidth = 1)
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("True Distribution and Reconstructed Distribution")
            
            legend_elements = [
                mpatches.Patch(facecolor = "#3b4cc0", edgecolor = "#b20326", label = "Reconstructed Distribution"),
                mlines.Line2D([], [], color = "red", linestyle = "--", label = "x1 and x2 = 0"),
                mlines.Line2D([], [], color = "black", linestyle = "-", label = "True Distribution")
            ]
            plt.legend(handles = legend_elements)
            
            plt.tight_layout()
            plt.savefig("Density_plot.png")
            print("Plot saved as Density_plot.png")
            
            #---------------------------------------- Loss plot
            plt.figure(figsize = (16, 8))
            
            plt.plot(result["lossvals"], color = "red", linewidth = 1)
            plt.axhline(0.05, color = "blue", linestyle = "--", linewidth = 1, label = "Loss = 0.05")
            plt.xlabel("Iteration")
            plt.ylabel("Loss Value")
            plt.title("Loss Value over each Iteration")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("Loss_plot.png")
            print("Plot saved as Loss_plot.png")
            
            
            # Stats reporting based on scatter samples
            report_statistics(characteristic_function, params, Z_reconstructed)
    
    parameters = define_topology(dimension, topology)
    weights = parameters["weights"]
    biases = parameters["biases"]
    result = execute(samples_count, frequencies_count, dimension, weights, biases, activations, resample_factor, loss, loss_lambda_parameter, lr, passes, batchsize, printLoss)
    createplots(makeplots)
    
    return result
    
#-------------------- Parameter Legend --------------------#
# mvnormal
# [
# [Mean vector] {dim = 2,1},
# [Covariance matrix] {dim = 2,2},
# ]

# mvnormal_mix
# [
# [Mean vector 1] {dim = 2,1},
# [Covariance matrix 1] {dim = 2,2},
# [Mean vector 2] {dim = 2,1},
# [Covariance matrix 2] {dim = 2,2},
# Mixture weight ( < 1 ) {dim = 1}
# ]

# eq_mvNIG_mixture
# [
# [Mean1 vector] (location) {dim = 2,1},
# [Beta1 vector] (skewness) {dim = 2,1},
# [Covariance1 matrix] (scale and correlation) {dim = 2,2},
# alpha1 scalar (tail > alpha^2 > beta.T@Cov@beta) {dim = 1},
# delta1 scalar (dispersion > 0) {dim = 1}
# [Mean2 vector] (location) {dim = 2,1},
# [Beta2 vector] (skewness) {dim = 2,1},
# [Covariance2 matrix] (scale and correlation) {dim = 2,2},
# alpha2 scalar (tail > alpha^2 > beta.T@Cov@beta) {dim = 1},
# delta2 scalar (dispersion > 0) {dim = 1}
# ]
#-------------------- Parameter Legend --------------------#

#-------------------- Notable Hyper-parameters --------------------#
# 25, 10, 2
# 300, 50, 2
# 5000, 50, 2
#-------------------- Notable Hyper-parameters --------------------#
