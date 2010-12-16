classdef Word < handle
    properties
        N     =  3; % number of states
        A     = []; % NxN transition probability matrix
        prior = []; % Nx1 initial state distribution vector
        mu    = []; % DxN mean vector (D = number of features)
        Sigma = []; % DxDxN covariance matrix
        name  = '';
    end
    
    methods
        function self = Word(name)
            self.name = char(name);
        end
        
        function log_likelihood = log_likelihood(self, observations)
            B = self.state_likelihood(observations);
            log_likelihood = forward(self, observations, B);
        end
        
        function [log_likelihood, alpha] = forward(self, observations, B)
            log_likelihood = 0;
            T = size(observations, 2);
            alpha = zeros(self.N, T);
            
            for t = 1:T
                if t == 1
                    % Initialization
                    alpha(:, t) = B(:, t) .* self.prior;
                else
                    % Induction
                    alpha(:, t) = B(:, t) .* (self.A' * alpha(:, t - 1));
                end
                
                % Scaling
                alpha_sum      = sum(alpha(:, t));
                alpha(:, t)    = alpha(:, t) ./ alpha_sum;
                log_likelihood = log_likelihood + log(alpha_sum);
            end
        end
        
        function beta = backward(self, observations, B)
            T = size(observations, 2);
            beta = zeros(self.N, T);
            
            % Initialization
            beta(:, T) = ones(self.N, 1);
            
            for t = (T - 1):-1:1
                % Induction
                beta(:, t) = self.A * (beta(:, t + 1) .* B(:, t + 1));
                
                % Scaling
                beta(:, t) = beta(:, t) ./ sum(beta(:, t));
            end
        end
        
        % Evaluates the Gaussian pdfs for each state at the observations
        % Returns a matrix containing B(s, t) = f(O_t | S_t = s)
        function B = state_likelihood(self, observations)
            B = zeros(self.N, size(observations, 2));
            
            for s = 1:self.N                
                B(s, :) = mvnpdf(observations', self.mu(:, s)', self.Sigma(:, :, s));
            end
        end
        
        function em_initialize(self, observations)
            % Random guessing
            self.prior = normalise(rand(self.N, 1));
            self.A     = mk_stochastic(rand(self.N));
            
            % All states start out with the empirical diagonal covariance
            self.Sigma = repmat(diag(diag(cov(observations'))), [1 1 self.N]);
            
            % Initialize each mean to a random data point
            indices = randperm(size(observations, 2));
            self.mu = observations(:, indices(1:self.N));
        end
        
        function train(self, observations)
            self.em_initialize(observations);

            for i = 1:15
                log_likelihood = self.em_step(observations);
                display(sprintf('Step %02d: log_likelihood = %f', i, log_likelihood))
                self.plot_gaussians(observations);
            end
        end
        
        function log_likelihood = em_step(self, observations)
            B = self.state_likelihood(observations);
            D = size(observations, 1);
            T = size(observations, 2);
            
            [log_likelihood, alpha] = self.forward(observations, B);
            beta                    = self.backward(observations, B);
            
            xi_sum = zeros(self.N, self.N);
            gamma  = zeros(self.N, T);
            
            for t = 1:(T - 1)
                % The normalizations are done to get valid distributions for each time step
                xi_sum      = xi_sum + normalise(self.A .* (alpha(:, t) * (beta(:, t + 1) .* B(:, t + 1))'));
                gamma(:, t) = normalise(alpha(:, t) .* beta(:, t));
            end
            
            gamma(:, T) = normalise(alpha(:, T) .* beta(:, T));
            
            expected_prior = gamma(:, 1);
            expected_A     = mk_stochastic(xi_sum);
            
            expected_mu    = zeros(D, self.N);
            expected_Sigma = zeros(D, D, self.N);
            
            gamma_state_sum = sum(gamma, 2);
            
            % Set any zeroes to one before dividing to avoid NaN
            gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0);
            
            for s = 1:self.N
                gamma_observations = observations .* repmat(gamma(s, :), [D 1]);
                expected_mu(:, s)  = sum(gamma_observations, 2) / gamma_state_sum(s);
                
                % Using Sigma = E(X * X') - mu * mu'
                % Also make sure it's symmetric
                expected_Sigma(:, :, s) = symmetrize(gamma_observations * observations' / gamma_state_sum(s) - ...
                                                     expected_mu(:, s) * expected_mu(:, s)');
            end
            
            % Ninja trick to ensure positive semidefiniteness
            expected_Sigma = expected_Sigma + repmat(0.01 * eye(D, D), [1 1 self.N]);
            
            % M-step
            self.prior = expected_prior;
            self.A     = expected_A;
            self.mu    = expected_mu;
            self.Sigma = expected_Sigma;
        end
        
        function plot_gaussians(self, observations)
            % Plotting two first dimensions
            
            plot(observations(1, :), observations(2, :), 'g+')
            hold on
            plot(self.mu(1, :), self.mu(2, :), 'r*')

            for s = 1:size(self.Sigma, 3)
                error_ellipse(self.Sigma(1:2, 1:2, s), 'mu', self.mu(1:2, s), 'style', 'r-', 'conf', .75)
            end

            axis([0 4000 0 4000])
            hold off
            title(sprintf('Training %s', self.name))
            xlabel('F1 [Hz]')
            ylabel('F2 [Hz]')
            drawnow
            
            %pause
        end
    end
end