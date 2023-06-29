import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from constants import DONATION_BOX_CAPACITY, SURVIVAL_THRESHOLD

font = {'family' : 'arial',
        'size'   : 20}

matplotlib.rc('font', **font)

def create_graphic(how_many=20, how_much=400, pool_limit=0, agent_threshold=0):
    number_of_agents = 2
    length = how_much
    apple_evos = np.zeros((number_of_agents, how_many, length))
    donation_evos = np.zeros((number_of_agents, how_many, length))
    pool_evos = np.zeros((how_many, length))

    inequality = np.zeros(how_many)
    apple_std = [0, 0]



    for it in range(how_many):

        for ag in range(number_of_agents):
            apple_evos[ag][it] = np.load("apple_evo" + str(ag) + "i" + str(it) +".npy")
            #os.remove("apple_evo" + str(ag) + "i" + str(it) +".npy")

            donation_evos[ag][it] = np.load("apple_donation" + str(ag) + "i" + str(it) +".npy")
            #os.remove("apple_donation" + str(ag) + "i" + str(it) +".npy")
        pool_evos[it] = np.load("pool_evoi"+ str(it) +".npy")
        #os.remove("pool_evoi"+ str(it) +".npy")

        inequality[it] = apple_evos[0][it][-1]/(apple_evos[1][it][-1] + apple_evos[0][it][-1])
        print(inequality[it])

    print(np.mean(inequality), np.std(inequality))
    ref = range(len(pool_evos[0]))

    apple_evo = [0, 0]
    donation_evo = [0, 0]
    pool_evolution = np.mean(pool_evos, axis=0)
    pool_std = np.std(pool_evos, axis=0)

    for ag in range(2):
        apple_evo[ag] = np.mean(apple_evos[ag], axis=0)
        apple_std[ag] = np.std(apple_evos[ag], axis=0)
        donation_evo[ag] = np.mean(donation_evos[ag], axis=0)

    donation_points = list()
    donation_points_corresponding = list()

    donated_points = list()
    donated_points_corresponding = list()
    taken_points = list()
    taken_points_corresponding = list()

    for i in range(len(donation_evo[0]) - 1):
        current = donation_evo[0][i]
        next = donation_evo[0][i+1]

        if next > current:
            print("Agent 1 donated here!", i+1)
            taken_points.append(i)
            taken_points_corresponding.append(apple_evo[0][i]+0.5)

    for i in range(len(donation_evo[1]) - 1):
        current = donation_evo[1][i]
        next = donation_evo[1][i+1]

        if next > current:
            print("Agent 2 donated here!", i+1)
            donation_points.append(i)
            donation_points_corresponding.append(apple_evo[1][i]+0.5)

    for i in range(len(donation_evo[0]) - 1):
        current = donation_evo[0][i]
        next = donation_evo[0][i+1]

        if next < current:
            print("Agent 1 took donation here!", i+1)
            donated_points.append(i)
            donated_points_corresponding.append(apple_evo[0][i])

    survival_rate = 0.0
    for i in range(how_many):
        true_1 = apple_evos[0][i][-1] >= SURVIVAL_THRESHOLD
        true_2 = apple_evos[1][i][-1] >= SURVIVAL_THRESHOLD

        if true_1 and true_2:
            survival_rate += 1

    print("How many times agents survived:", survival_rate / how_many)


    print("\\\\ \hline \multicolumn{1}{|c|}{$K = " + str(SURVIVAL_THRESHOLD) + ", C = " + str(DONATION_BOX_CAPACITY) + "$}  ")
    part_agent_1 = " & \multicolumn{1}{c|}{$ " + str(round(apple_evo[0][-1],1)) + " \pm" + str(round(apple_std[0][-1], 1))  + "$}"
    part_agent_2 = " & \multicolumn{1}{c|}{$ " + str(round(apple_evo[1][-1],1)) + " \pm " + str(round(apple_std[1][-1],1))  + "$}"
    part_pool = " & \multicolumn{1}{c|}{$ " + str(round(pool_evolution[-1],1)) + " \pm " + str(round(pool_std[-1], 1))  + "$}"
    print(part_agent_1 + part_agent_2 + part_pool)

    fig, ax = plt.subplots()

    everywhen = 399

    plt.axhline(y=SURVIVAL_THRESHOLD, color="black", linestyle="--")
    if DONATION_BOX_CAPACITY < 100:
        plt.axhline(y=DONATION_BOX_CAPACITY, color="black", linestyle=":")
    plt.plot(ref, apple_evo[0], marker="d", markevery=everywhen, color="blue", label="Agent 1",markersize=10)

    plt.plot(ref, apple_evo[1], marker="o", markevery=everywhen, color="orange", label="Agent 2",markersize=10)

    #for i in donation_points:
        #print(i)
        #plt.axvline(x=i, color="orange", alpha=0.4, ymin=0.35, ymax=0.45)

    #for i in range(len(taken_points)):
        #print(i)
        #plt.axvline(x=taken_points[i], color="blue", alpha=0.4, ymin=0.35, ymax=0.45)

    #for i in range(len(donated_points)):
        #print(donated_points_corresponding[i])
        #plt.axvline(x=donated_points[i], color="blue", alpha=0.4, ymax=donated_points_corresponding[i]/26)

    #plt.scatter(donation_points, donation_points_corresponding, color='lightcoral', marker='^',edgecolors='black', s=90)
    #plt.scatter(donated_points, donated_points_corresponding, color="blue", marker='v',edgecolors='black', s=90)
    #plt.scatter(taken_points, taken_points_corresponding, color="blue", marker='^',edgecolors='black', s=90)
    ax.fill_between(ref, apple_evo[0] + apple_std[0], apple_evo[0] - apple_std[0], alpha=0.2)
    ax.fill_between(ref, apple_evo[1] + apple_std[1], apple_evo[1] - apple_std[1], alpha=0.2)

    plt.plot(ref, pool_evolution, marker="s", markevery=everywhen, color="green", label="Donation box", markersize=10)
    #plt.xlabel("Time-step")
    #plt.ylabel("Apples")
    #ax.xaxis.set_label_position('top')
    ax.fill_between(ref, pool_evolution + pool_std, pool_evolution - pool_std, alpha=0.2)

    # show a legend on the plot
    plt.legend(loc=2)
    # Display a figure.
    #plt.show()

    what="ethical"

    plt.ylim(-2, 32)
    #plt.ylim(-8, 12)
    #plt.savefig('ExperimentK'+str(agent_threshold)+"C"+str(pool_limit)+what+'.png')
    plt.show()

if __name__ == '__main__':
    create_graphic(how_many=1, how_much=400, pool_limit=DONATION_BOX_CAPACITY, agent_threshold=SURVIVAL_THRESHOLD)
