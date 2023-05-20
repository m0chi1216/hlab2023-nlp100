def temp(x,y,z):
    return str(f'{x}時のとき{y}は{z}')

x=12
y="気温"
z=22.4

print(temp(x,y,z))


def generate_sentence(x, y, z):
  print(f'{x}時のとき{y}は{z}')

generate_sentence(12, '気温', 22.4)