#export
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import emoji
import numpy as np
import seaborn as sns


def convert(year,month,date):
    return int(datetime(year, month, date, 0, 0, 0).timestamp()*1000)

def convert_reverse(timestamp):
    dt_object = datetime.fromtimestamp(timestamp/1000)
    print("dt_object =", dt_object)
    return dt_object

def add_extra_timeperiod(plt,window,start_time,end_time):
    start_election = convert(2019,4,11)
    end_election = convert(2019,5,23)
    pulwama_event= convert(2019,2,14)
    balakot_event= convert(2019,2,27)


    day_start=int((start_election-start_time)/(window*24*60*60*1000))   
    day_end=int((end_election-start_time)/(window*24*60*60*1000))   
    plt.axvline(day_start, linestyle='--', color='r')
    for i in range(day_start+1,day_end):
        plt.axvline(i, linestyle='--',alpha=0.2, color='r')
    plt.axvline(day_end, linestyle='--', color='r')

    day_pulwama=int((pulwama_event-start_time)/(window*24*60*60*1000))
    day_balakot=int((balakot_event-start_time)/(window*24*60*60*1000))

    plt.axvline(day_pulwama, linestyle='--',alpha=0.5, color='k',linewidth=1.5)
    plt.axvline(day_balakot, linestyle='--',alpha=0.5, color='k',linewidth=1.5)


    x_tick_keys=[]
    x_tick_label=[]

    for year in [2018,2019]:
        for month in range(1,13):
            if(year==2019 and (month in [3,4,6])):
                continue

            timestamp_begin_month=convert(year,month,1)
            if(timestamp_begin_month>start_time and timestamp_begin_month<end_time):
                first_day_month =int((timestamp_begin_month-start_time)/(window*24*60*60*1000))
                x_tick_keys.append(first_day_month)
                x_tick_label.append(str(month)+'/'+str(year))


    #### for pulwama +balakot
    x_tick_keys.append(day_pulwama)
    x_tick_label.append('pulwama event')

    x_tick_keys.append(day_balakot)
    x_tick_label.append('balakot event')

    x_tick_keys.append(day_start)
    x_tick_label.append('election start')

    x_tick_keys.append(day_end)
    x_tick_label.append('election end')
    plt.xticks(x_tick_keys, x_tick_label, rotation=90)
    return plt




def emoji_string(ele):
    emoji_files=glob.glob("../Political_Results/Emoji/*")
    try:
        emoji_name="-".join(emoji.demojize(ele)[1:-1].split('_'))
        
    except:
        return " "
    str1= " "
    for emoji_file in emoji_files:
        temp=emoji_file.split("/")[3]
        emoji_filename = temp.split("_")[0]
        if(emoji_filename==emoji_name):
            return "\includegraphics[height=1em]{samples/Emoji/"+temp+"}"
    print(emoji_name+"  not found")
    return str1   
    
def latex_emoji_communities(community_dict):

    str1="\\begin{table}[!htb]\n\\begin{tabular}{c}\n \hline Emojis \\\\\hline"
    for key in community_dict:
        for ele in community_dict[key]:
                str1+=emoji_string(ele)+","
        str1+="\\\\\\hline\n"
    str1+="\end{tabular}\n\caption{Co-occuring emojis captured as communities}\n \label{tab:emoji_communities}\n\end{table}"
    return str1




def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)