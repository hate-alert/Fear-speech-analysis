#export
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import emoji
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