'''get stopwords'''
import pandas as pd
#stopwords already saved from NLTK library in case of internet problem
stopwords=pd.read_csv('stopwords.csv')['sw']
swl=set(stopwords.tolist())
'''unique names of sentiment and the counts of each'''
import pandas as pd
allSt=pd.read_csv('./tweet_emotions.csv')
sentiment_counts=allSt["sentiment"].value_counts()
'''split'''
def spl(allSt):
    allSt['content']=allSt['content'].str.replace(r'\@\S+\s','', regex=True).str.replace(r'[^A-Za-z\s\d]','', regex=True).str.lower().apply(lambda x: [word for word in x.split() if word not in swl])
    pf=pd.DataFrame({'word':allSt['content'].explode()}).join(allSt.drop('content',axis=1))
    dic={}
    for i in allSt["sentiment"].unique():
        dic[i]=pf[pf['sentiment']==i]
    ct={sen:df['word'].value_counts() for sen,df in dic.items()}
    return ct
ct=spl(allSt)
'''normalize (preparation for filtering out common words with no emotional inclinations)'''
def nor(ct):
    norCt={}
    for sen,freqs in ct.items():
        minFreq=freqs.min()
        maxFreq=freqs.max()
        norFreqs=((freqs-minFreq)/(maxFreq-minFreq))*100
        #100/(maxFreq-minFreq) is the normalization ratio, because the normalization range is [0,100]
        norCt[sen]=norFreqs
    return norCt
norCt=nor(ct)
'''filter out common words'''
def com(norCt,ct):
    combined=pd.concat(norCt.values()).apply(lambda x:(int(x) if type(x)==float else x))
    ctDf=pd.concat(ct,axis=1)
    stat=combined.groupby(combined.index).agg(['min','max','count',lambda x:x.mode().iloc[0] if not x.mode().empty else None]).rename(columns={'<lambda_0>':'mode'})
    toRem=stat[(stat['count']>=5)&(((stat['max']-stat['min'])<=10)|((stat['max']-stat['mode'])<=10))].index
    #if the frequency is not less than 5 and any value within (range,max-mode-distance) of its counts is not greater than 10, then filter this word out from all ctwords lists
    ctDf=ctDf.drop(index=toRem,errors='ignore')
    ct={sen:ctDf[sen].dropna().apply(lambda x:(int(x) if type(x)==float else x)).sort_values(ascending=False).head(10) for sen in ctDf.columns}
    #print('Removed:',toRem)
    return ct
result=com(norCt,ct)
'''plot'''
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
import os
# Define sentiment colors and their brightness
emotionColors = {
    'neutral': ('green', 'dark'),
    'worry': ('gray', 'dark'),
    'happiness': ('yellow', 'light'),
    'sadness': ('darkblue', 'dark'),
    'love': ('pink', 'light'),
    'surprise': ('gold', 'light'),
    'fun': ('purple', 'dark'),
    'relief': ('lightblue', 'light'),
    'hate': ('black', 'dark'),
    'enthusiasm': ('orange', 'light'),
    'boredom': ('brown', 'dark'),
    'anger': ('red', 'dark')
}
# Function: Convert hexadecimal color to RGB
def hexToRgb(hexColor):
    if hexColor.startswith('#'):
        return tuple(int(hexColor[i:i+2], 16) for i in (1, 3, 5))
    else:
        return mcolors.to_rgb(hexColor)
# Create .imgs directory if not exists
if not os.path.exists('.imgs'):
    os.makedirs('.imgs')
# Generate pie chart and word cloud and combine them
for sentiment, words in result.items():
    # Generate pie chart
    figPie, axPie = plt.subplots(figsize=(6, 6))
    color, lightness = emotionColors.get(sentiment, ('gray', 'dark'))
    topWords = words.head(10)
    labels = [f"{word}: {count}" for word, count in topWords.items()]
    # Convert color to RGB
    rgbColor = hexToRgb(color)
    # Determine text color based on brightness
    textColor = 'white' if lightness == 'dark' else 'black'
    # Use wedgeprops parameter to adjust pie chart border width and font size
    wedges, texts, autotexts = axPie.pie(topWords, labels=labels, autopct='%1.1f%%', colors=[color] * len(topWords), startangle=90,
                                          wedgeprops=dict(edgecolor='white', linewidth=1),
                                          textprops=dict(size=14, color='black'))
    # Set percentage text color
    for autotext in autotexts:
        autotext.set_color(textColor)
    axPie.set_title(f'sentiment: {sentiment} - Top10', fontsize=16)
    plt.tight_layout()
    
    # Save pie chart to file (with overwrite)
    piePath = os.path.join('.imgs', f'{sentiment}_pie.png')
    if os.path.exists(piePath):
        os.remove(piePath)  # Remove existing file
    figPie.savefig(piePath, bbox_inches='tight')
    plt.close(figPie)
    
    # Generate word cloud
    wordFreq = dict(zip(topWords.index, topWords.values))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordFreq)
    figWordcloud, axWordcloud = plt.subplots(figsize=(8, 4))
    axWordcloud.imshow(wordcloud, interpolation='bilinear')
    axWordcloud.axis("off")
    axWordcloud.set_title(f'{sentiment} Sentiment Word Cloud', fontsize=16)
    plt.tight_layout()
    
    # Save word cloud to file (with overwrite)
    wordcloudPath = os.path.join('.imgs', f'{sentiment}_wordcloud.png')
    if os.path.exists(wordcloudPath):
        os.remove(wordcloudPath)  # Remove existing file
    figWordcloud.savefig(wordcloudPath, bbox_inches='tight')
    plt.close(figWordcloud)
    
    # Open pie chart and word cloud images
    pieImage = Image.open(piePath)
    wordcloudImage = Image.open(wordcloudPath)
    
    # Calculate combined image size
    totalWidth = pieImage.width + wordcloudImage.width
    maxHeight = pieImage.height
    
    # Create new image
    combinedImage = Image.new('RGB', (totalWidth, maxHeight), 'white')
    
    # Paste pie chart
    combinedImage.paste(pieImage, (0, 0))
    
    # Calculate word cloud vertical center position
    wordcloudY = (maxHeight - wordcloudImage.height) // 2
    
    # Paste word cloud
    combinedImage.paste(wordcloudImage, (pieImage.width, wordcloudY))
    
    # Save combined image (with overwrite)
    combinedPath = os.path.join('.imgs', f'- {sentiment}.png')
    if os.path.exists(combinedPath):
        os.remove(combinedPath)  # Remove existing file
    combinedImage.save(combinedPath)
    
    # Close image files
    pieImage.close()
    wordcloudImage.close()
    combinedImage.close()
    
    # Delete individual files after combining
    if os.path.exists(piePath):
        os.remove(piePath)
    if os.path.exists(wordcloudPath):
        os.remove(wordcloudPath)
print("All combined images have been generated and saved in the .imgs directory. Individual pie charts and word clouds have been deleted.")