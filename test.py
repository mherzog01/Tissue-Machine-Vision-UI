import numpy as np

#v = np.random.randint(0,10,5)
#c = np.random.randint(0,2,5)
v = np.array([8, 7, 8, 9, 8])
c = np.array([1, 0, 0, 1, 0])
print(v)
print(c)

i = np.arange(len(v))
print(i)
#z = [(e0,e1) for (e0,e1) in zip(c,v)]
#print(z)
c_0 = (c == 0)
print(v[c_0])
print(i[c_0])

# _, idx = np.unique(c, return_index=True)
# print(np.maximum.reduceat(v, idx))


# https://stackoverflow.com/a/50241538/11262633
# mask = c_0
# ind_local = np.argmax(v[mask])
# G = np.ravel_multi_index(np.where(mask), mask.shape)
# ind_global = np.unravel_index(G[ind_local], mask.shape)
# print(ind_global)

mask = (c == 0) & (v >= 8)
if not mask.any():
    print('No matching')
else:
    ind = np.arange(len(c))
    ind_local = np.argmax(v[mask])
    ind_global = ind[mask][ind_local]
    print(ind_global, v[ind_global])
