from .linear_cost_functions import mse,\
                                    rmse,\
                                    mae
from .cross_entropy_functions import binary_cross_entropy,\
                                        cross_entropy

linear_cost_function_methods = {'MSE': mse,\
                                'RMSE': rmse,\
                                'MAE': mae}
