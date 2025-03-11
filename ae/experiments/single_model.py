import matplotlib.pyplot as plt
from ae.experiments.training import Trainer

num_test = 1000
device = "cpu"
# Define a list of time horizons to test
# Load the pre-trained model
model_dir = "ae/experiments/trained_models/Paraboloid/Lang/trained_20250310-225959_h[32]_df[16]_dr[16]_lr0.001_epochs9000_not_annealed"
trainer = Trainer.load_from_pretrained(model_dir)
print(trainer.models.keys())

for name, model in trainer.models.items():
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    trainer.toy_data.set_point_cloud(1.)
    x = trainer.toy_data.generate_data(num_test, 2)["x"]
    model.autoencoder.plot_surface(-1, 1, 30, ax=ax, title=name)
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    plt.show()
