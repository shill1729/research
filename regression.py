import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)


# ----- Data Generation -----
def generate_data(f, df, n_train=100, n_test=100, noise_std=0.01):
    x_train = torch.linspace(-2, 2, n_train).unsqueeze(1)
    y_train = f(x_train) + noise_std * torch.randn_like(x_train)
    dy_train = df(x_train)

    x_test = torch.linspace(-2.5, 2.5, n_test).unsqueeze(1)
    y_test = f(x_test)
    return x_train, y_train, dy_train, x_test, y_test


# ----- Model -----
class FFNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


# ----- Loss computation -----
def compute_derivative(model, x):
    x = x.clone().detach().requires_grad_()
    y = model(x)
    dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=False)[0]
    return dydx.detach()


def train(model, x_train, y_train, dy_train=None, lambda_deriv=0.0, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()

        x_input = x_train.clone().detach().requires_grad_(lambda_deriv > 0.0)
        y_pred = model(x_input)
        loss = nn.functional.mse_loss(y_pred, y_train)

        if dy_train is not None and lambda_deriv > 0.0:
            dydx_pred = torch.autograd.grad(
                y_pred, x_input,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            loss += lambda_deriv * nn.functional.mse_loss(dydx_pred, dy_train)

        loss.backward()
        optimizer.step()



# ----- Evaluation -----
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_mse = nn.functional.mse_loss(y_pred, y_test)
    return test_mse.item()


# ----- Experiments -----
def run_experiment(f, df, name, lambda_deriv=1.0):
    x_train, y_train, dy_train, x_test, y_test = generate_data(f, df)

    model_vanilla = FFNN()
    train(model_vanilla, x_train, y_train, dy_train=None, lambda_deriv=0.0)
    err_vanilla = evaluate(model_vanilla, x_test, y_test)

    model_tangent = FFNN()
    train(model_tangent, x_train, y_train, dy_train=dy_train, lambda_deriv=lambda_deriv)
    err_tangent = evaluate(model_tangent, x_test, y_test)

    print(f"Function: {name}")
    print(f"  Vanilla Test MSE:   {err_vanilla:.6f}")
    print(f"  Tangent Test MSE:   {err_tangent:.6f}")

    with torch.no_grad():
        y_vanilla = model_vanilla(x_test)
        y_tangent = model_tangent(x_test)
    plt.figure(figsize=(6, 4))
    plt.plot(x_test.numpy(), y_test.numpy(), label='True f', c='k')
    plt.plot(x_test.numpy(), y_vanilla.numpy(), label='Vanilla NN', ls='--')
    plt.plot(x_test.numpy(), y_tangent.numpy(), label='Tangent NN', ls=':')
    plt.title(f"{name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----- Run -----
if __name__ == "__main__":
    funcs = [
        (lambda x: torch.sin(x), lambda x: torch.cos(x), "sin(x)"),
        (lambda x: x ** 2, lambda x: 2 * x, "x^2"),
        (lambda x: x ** 3, lambda x: 3 * x ** 2, "x^3")
    ]
    for f, df, name in funcs:
        run_experiment(f, df, name, lambda_deriv=1.0)
