# *_*coding:utf-8 *_*
from scipy.interpolate import interp1d

from utils import *

if __name__ == "__main__":

    # get distance data, if there is a doc for that, we read it; if not, we calculate
    # it with the help of the images
    if os.path.isfile(path_distance):
        distance = get_data(path_distance)[:130]
    else:
        distance = get_distance_all(path, path_save)[:130]

    # if there is no .gif, we generate a .gif to show how the points move
    if not os.path.isfile(os.path.join(path_save, name + '.gif')):
        generate_gif(path_save, name + '.gif')

    # loading position data(global)
    global_pos_data = load_pos()
    global_time = global_pos_data["t(s)"][:-5]
    global_pos = global_pos_data[" position1"][:-5]
    global_pos /= global_pos[0]

    # loading force data
    force_data = get_data(path_force)
    temps = force_data["t(s)"]
    force = force_data[" F(N)"]

    # interpolation for force data
    func_f = interp1d(temps, force)

    # get force with  corresponding global position
    force_c_global = func_f(global_time)
    # get force with  corresponding local distance
    force_c_local = func_f(distance["t"])

    # plot function
    plt.title("contraintes - deformation (" + name + ')')
    plt.xlabel("deformation(mm/mm)")
    plt.ylabel("contraintes(MPa)")
    plt.plot(global_pos - 1, force_c_global / 2.5 / 6, label="global longitudinal")
    plt.plot(distance["x"] - 1, force_c_local / 2.5 / 6, 'y.', label='local longitudinal')
    plt.plot(distance["y"] - 1, force_c_local / 2.5 / 6, 'g.', label='local transversal')
    force_curve_1 = optimize_func(func, distance["x"] - 1, force_c_local / 2.5 / 6)
    force_curve_2 = optimize_func(func, distance["y"] - 1, force_c_local / 2.5 / 6)
    plt.plot(distance["x"] - 1, force_curve_1, 'gray', label='fit local longitudinal')
    plt.plot(distance["y"] - 1, force_curve_2, 'b', label='fit local transversal')
    plt.legend()
    plt.savefig(os.path.join('data', name + 'fin.png'))
    plt.show()
