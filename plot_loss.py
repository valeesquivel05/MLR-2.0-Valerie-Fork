import matplotlib.pyplot as plt
from torch import load

# plot loss data from training
loss_data = load('mvae_loss_data_recurr.pt')
plt.plot(loss_data['retinal_train'], label='Retinal Training Error')
plt.plot(loss_data['retinal_test'], label='Retinal Test Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.legend()
plt.title('Retinal Loss Over Epochs of Training')
plt.show()

plt.plot(loss_data['cropped_train'], label='Cropped Training Error')
plt.plot(loss_data['cropped_test'], label='Cropped Test Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.legend()
plt.title('Cropped Loss Over Epochs of Training')
plt.show()