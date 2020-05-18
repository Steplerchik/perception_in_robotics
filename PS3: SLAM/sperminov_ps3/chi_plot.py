import pickle
from matplotlib import pyplot as plt

with open("output_data.txt", "rb") as fp:   # Unpickling
    chi_squared_error = pickle.load(fp)
print(chi_squared_error.shape)

plt.ylim(0, 1000)
plt.plot(chi_squared_error)
plt.title('Chi squared error')
plt.xlabel('t')
plt.show()
