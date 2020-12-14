import torch
import time
import opt.nlsdp_ipm as ipm


def train_model(model, loaders=None, method="ipm", options=None,
                constraints=None, mse_type='mean', obj="sim"):

    print("Beginnning training: {}".format(time.ctime()))
    loss = torch.nn.MSELoss(reduction=mse_type)

    # Generate optimization problem
    options = ipm.make_default_options() if options is None else options
    problem = ipm.nlsdp(model=model, criterion=loss,
                        train_loader=loaders["Training"], **options)

    # Add constraints to problem
    if constraints is not None:

        # SDP constraints
        if constraints["lmi"] is not None:
            for lmi in constraints["lmi"]:
                problem.addSDPconstraint(lmi)

        # Inequality Constraints
        if constraints["inequality"] is not None:
            for ineq in constraints["inequality"]:
                problem.addInequality(ineq)

    log, best_model = problem.solve()
    return log, best_model
