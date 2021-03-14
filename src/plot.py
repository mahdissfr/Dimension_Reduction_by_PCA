from matplotlib.pyplot import figure, show



def get_ES_result_in_2D(chromosome):
    xp = []
    yp = []
    z = chromosome.get_z_array()
    a,b= chromosome.get_normal_ab()
    for i in range(len(z)):
        xp.append(a*z[i])
        yp.append(b*z[i])
    return xp, yp


def plot(chromosome):
    """
    Plot data points with the best vector for dimension reduction
    :return:
    """
    xp, yp= get_ES_result_in_2D(chromosome)

    fig = figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.scatter(chromosome.x[0:chromosome.chromosome_length], chromosome.y[0:chromosome.chromosome_length], label="xy", color="black",marker="x")
    ax.tick_params(axis='x', colors="black")
    ax.tick_params(axis='y', colors="black")

    ax2.plot(xp, yp, '-o',label="z", color='red', marker='o', markersize=3, markerfacecolor='blue', linestyle='dashed', linewidth=0.1)
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', colors="grey")
    ax2.tick_params(axis='y', colors="grey")

    show()
