import scipy.io

data = scipy.io.loadmat("tapping_data/PDBS13_1.mat")

print("Keys:", data.keys())

for key in data:
    if not key.startswith("__"):
        print("\nKey:", key)
        print("Type:", type(data[key]))
        try:
            print("Shape:", data[key].shape)
        except:
            pass