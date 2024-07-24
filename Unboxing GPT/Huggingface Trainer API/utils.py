import matplotlib.pyplot as plt

def plot_loss_curve(losses, val_losses):
    plt.plot(losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')