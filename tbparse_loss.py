from tbparse import SummaryReader
from matplotlib.figure import  Figure

log_dir = "tensorboard"

reader = SummaryReader(log_dir)
df = reader.scalars

loss_df = df[df["tag"]=="training_loss"]

step_arr = loss_df.step.values
loss_arr = loss_df.value.values

fig = Figure()
ax = fig.subplots()

ax.set_ylim(bottom=0.00001, top=1)
ax.set_yscale('log')
ax.plot(step_arr, loss_arr)

fig.savefig("loss.png")
