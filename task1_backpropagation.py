from pathlib import Path
import matplotlib.pyplot as plt


x = 2
t = 10
w = 1
b = 0


def calculate_loss(w, b):
    y = (w * x) + b
    loss = (y - t) ** 2
    return loss


y = (w * x) + b
error = y - t

dw = 2 * error * x
db = 2 * error

print("========== HAND CALCULATED GRADIENTS ==========")
print("Initial Loss:", calculate_loss(w, b))
print("dL/dw:", dw)
print("dL/db:", db)


def gradient_descent(lr, steps=50, show_output=False):

    w = 1
    b = 0

    losses = []

    for step in range(1, steps + 1):

        y = (w * x) + b

        loss = (y - t) ** 2

        losses.append(loss)

        error = y - t

        dw = 2 * error * x
        db = 2 * error

        w = w - lr * dw
        b = b - lr * db

        if show_output and step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.10e}")

    final_loss = calculate_loss(w, b)

    losses.append(final_loss)

    return losses



print("\n========== GRADIENT DESCENT ==========")

losses_005 = gradient_descent(
    lr=0.05,
    steps=50,
    show_output=True
)


print("\n========== LEARNING RATE COMPARISON ==========")

learning_rates = [0.001, 0.05, 0.5]

all_losses = {}

for lr in learning_rates:

    losses = gradient_descent(
        lr=lr,
        steps=50
    )

    all_losses[lr] = losses

    print(
        f"Learning Rate: {lr} | "
        f"Initial Loss: {losses[0]:.6f} | "
        f"Final Loss: {losses[-1]:.6e}"
    )



plt.figure(figsize=(10, 6))

for lr, losses in all_losses.items():

    plt.plot(
        range(len(losses)),
        losses,
        label=f"Learning Rate = {lr}"
    )

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Gradient Descent - Learning Rate Comparison")

plt.yscale("log")

plt.legend()
plt.grid(True)

file_path = Path(__file__).resolve().parent / "task1_loss_curve.png"

plt.savefig(file_path, dpi=300)

print("Graph saved at:", file_path)

plt.show()

print("\nLoss curve saved successfully!")
