import matplotlib.pyplot as plt
import time
import pandas as pd

def plot_result(date_array, date_price_array, degree_of_truth_array):
    plt.clf()
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400

    # main plot
    plt.scatter(date_array, date_price_array, c=degree_of_truth_array, cmap=plt.cm.Reds, edgecolor='none', s=6)

    # Legend
    length = int(len(date_array)/10)
    a = date_array[:length+1]
    b = [max(date_price_array) for _ in range(length+1)]
    c = [i/length for i in range(length+1)]
    plt.scatter(a, b, c=c, cmap=plt.cm.Reds, edgecolor='none', s=20)

    # text
    txt = ["low risk", "high risk"]
    for i, v in enumerate([0, length]):
        plt.annotate(txt[i], xy=(a[v], b[v]), xytext=(a[v], b[v]*0.95), fontsize=8)

    pd.DataFrame(date_array).to_csv("date_array.csv")
    pd.DataFrame(date_price_array).to_csv("date_price_array.csv")
    pd.DataFrame(degree_of_truth_array).to_csv("degree_of_truth_array.csv")

    # label
    plt.title("DJIA Risk Prediction", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("DJIA Index", fontsize=14)

    # Save to file
    now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_name = 'plot_'+now_time+'.png'
    plt.savefig(file_name, bbox_inches='tight')

    # Show the plot
    plt.show()



if __name__=="__main__":
    date_array = [i for i in range(28)]
    date_price_array = [10, 20, 15, 100, 10, 20, 15, 100, 10, 20, 15, 100, 10, 20, 15, 100, 10, 20, 15, 100, 10, 20, 15, 100, 10, 20, 15, 100]
    degree_of_truth_array = [0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5, 0.2, 0.2, 0.99, 0.5]

    plot_result(date_array, date_price_array, degree_of_truth_array)