# Amazon Transcribe Sensitivity Analysis
Audio is a clear form of communication, but transcripts reinforce it. In this blog post, we would show how Amazon Transcribe is used to improve your experience of listening to music with lyrics. We start with an overview of the AWS Transcribe function, explained how various data and songs are collected and analyze data, and discussed real-word application of AWS Transcribe that facilitates our day to day.

Utilize AWS services (SageMaker, S3 Storage Bucket) to conduct Sensitivity Analysis on Amazon Transcribe Machine Learning Model

## Table of contents

- [AWS Transcribe Overview](#AWS-Transcribe-Overview)
- [Motivation & Aims of the Project](#Motivation-&-Aims-of-the-Project)
- [Analysis](#Analysis)
  * [Setup](#Setup)
  * [Steps of Using AWS Transcribe](#Steps-of-Using-AWS-Transcribe)
  * [Lyrics Fetch and Levenshtein Distance](#Lyrics-Fetch-and-Levenshtein-Distance)
  * [Production and Analysis](#Production-and-Analysis)
  * [Architecture Overview](#Architecture-Overview)
- [Conclusion & Discussions](#Conclusion-&-Discussions)
- [Citations](#citations)

# AWS Transcribe Overview
According to National Institute on Deafness and Other Communication Disorders [NIDCD](https://www.nidcd.nih.gov/health/statistics/quick-statistics-hearing), approximately 0.2 ~ 0.3% of children are born with a noticeable hearing loss level in one or both ears in the United States and about 37.5 million American adults report having trouble hearing. People with impaired hearing deserve a quality life and AWS Transcribe can help this out.
Amazon Transcribe utilizes a deep learning process called automatic speech recognition (ASR) to convert speech to text accurately and effectively. It analyzes audio files and uses advanced machine learning technologies to transcribe voice data into texts. It can be used to transcribe phone calls, automate subtitling, and generate metadata for media assets to create a fully searchable archive. (ref: [aws-transcribe](https://aws.amazon.com/transcribe/))

![transcribelogo](/images/transcribe.png)

# Motivation & Aims of the Project

When comparing Apple Music and Spotify, Apple music took the lead in its live lyrics feature, which provides an on-screen, karaoke-style feature. It is not until March 2021 did Spotify finally released the new lyrics feature on its platform, partnered with Genius and Musicmatch. However, as the new lyrics feature is still in its testing phase, it has not covered all the songs on the platform. Indeed, Spotify has been receiving complaints from users that they cannot access the lyrics completely. So, unfortunately, users have to wait for the new real-time lyrics feature to catch up with the great amount of music available on the platform.
Imagine a situation that you fall in love with a blue song in a foreign language and you are desperate to learn what it expresses, or, that you got ignited by a rap song but you can’t figure out what it’s rapping unless you have a real-time lyrics book on hand. In such situations, you would get mad staring at the blank lyrics page. That brings us the inspiration to generate lyrics on our own. Fortunately, with the help of Amazon Transcribe, we are able to achieve this goal.

![musicnerd](/images/nerd.jpeg)

The question that comes to mind spontaneously is that whether AWS Transcribe is capable of translating with high accuracy. Indeed, language is so delicately nuanced and contextual that it takes a lot of rules, reasoning, data, and processing power for people to understand. Spenser Mestel, writing in The Atlantic, points out that “Language is a data set that’s far more complex than it seems, so no matter how quickly translation technology evolves, the stochastic messiness of speech will always outpace it”. The limitation of machine translation is also experienced everywhere in daily life. When you watch a recording of a last week’s class that you missed, you may find that the professor speaks so fast with a tone that you just can’t get along with. However, when you turn on the auto-generated subtitle, you find it only makes things worse: API translated into “AP eyes” is just one of the numerous examples. So, imaginably, when translating songs with background instrumental “noises” and lyrics with speeds ranging from sighing to rapping, unexpected punctuation, and varying pitching, the performance of any machine learning translator could be very unpredictable. Therefore, in this project, through inputting numerous mp3 song files into the AWS Transcribe translator, we compare the generated lyrics with the correct lyrics, and summarize from the results what input factors influence the accuracy of AWS Transcribe’s work.
Among the factors of a song file that could influence the accuracy of Amazon Transcribe's work, there are sample rate and bit rate. In computer, digital audio is converted into information readable by a computer system. Certain characteristics of a sound wave, like the frequency and amplitude, are converted to binary data that computer software can read. Samples are the series of snapshots taken on the digital audio. A sample is taken at a particular time in the audio wave, recording amplitude. This information is then converted into binary data. The rate at which the samples are taken snapshot of is the sample rate, measured in kilohertz. Bit rate is, on the other hand, the number of bits in each sample, or, the rate at which bits are transferred from one location to another. More bits are used to represent the audio data for each second of playback, or, higher bit rate, usually corresponds to better sound quality.

![samplerate](/images/samplerate.png)

Accuracy and precision build credibility; so, it is vital to test how accurate AWS Transcribe is before applying it to music apps like Spotify. In order to test how accurately AWS Transcribe can transcribe songs to lyrics, we compute the Levenshtein Distance (LD) -- a number that informs people how similar two sequences are. The lower the given number is, the more similar the two sequences are. LD is named after Russian Scientist Vladimir Levenshtein who invented this algorithm in 1965. It is a popular and useful technique that has been used in fields like speech recognition, DNA analysis, and plagiarism detection. As we are comparing the generated lyrics with the correct lyrics, we will apply this algorithm as well. Matrices will be constructed to compute the LD; after that, we will analyze the numbers and determine whether AWS Transcribe can accurately generate lyrics.

# Analysis

## Setup

#### Initialization
Run _Initialize.ipynb_ first and follow the instructions for pre-analysis data preparation

#### Transcribe
Run _Transcribe.ipynb_ for audio transcription using Amazon Transcribe, and retrieve corresponding lyrics from the lyrics.ovh API. Check here for API [Documentation](https://lyricsovh.docs.apiary.io/#)

#### Sample Data
data.csv, test_copy.csv, and tempo.csv are provided.

------
data.csv - dataframe containing song title, artist, genre, lyrics, and transcribed lyrics

test_copy.csv - processed data.csv, containing transcription accuracy calculated through the Levenshtein Distance

tempo.csv - contains song title, artist, and tempo

## Steps of Using AWS Transcribe
1. Log in to your AWS account and select **amazon sagemaker**
![step1](/images/step1.JPG)
2. Click on **IAM role ARN** and you will see the following webpage
![step2](/images/step2.JPG)
3. Search for **transcribe**
![step3](/images/step3.JPG)
4. Select the policy named **AmazonTranscribeFullAccess** and attach it to your notebook instance
![step4](/images/step4.JPG)

## Lyrics Fetch and Levenshtein Distance
Here, we display how to find the Levenshtein Distance that represents the accuracy of the translated lyrics compared to the true lyrics. We iterated through all music files in google drive and dropped songs without metadata. We used the lyrics.ovh API to retrieve the lyrics for the songs and did some formatting to display the true lyrics and transcribed lyrics in the table.
Here is a table containing the information about the 632 songs as well as their true lyrics and transcribed lyrics.

![song](/images/Song.png)

Levenshtein ratio and distance are computed to test the accuracy of AWS Transcribe. We first created a matrix and populated it with the indices of every character from the two strings (true lyrics and generated lyrics). After that, we iterate over the matrix to compute the cost of deletion, insertion, and substitution. In order to align the results with those of the Python Levenshtein package, the cost of a substitution would be 2 if we choose to calculate the ratio and the cost of a substitution would be 1 if we merely calculate the distance.
We did two OLS Regression Models to test how bitrate, sample rate, tempo, and genre relates to the accuracy of AWS Transcribe. The first column of the figure below is the result of the first model, which contains Accuracy ~ Sample + Bitrate + Tempo. The second column is the result of the second model, which not only contains Accuracy ~ Sample + Bitrate + Tempo but also includes dummy-coded variable genre.

![genre](/images/Genre.png)

According to the result, we interpret that bitrate, sample rate, tempo, genre dance, and genre Hip Hop are positively correlated with accuracy when other covariates are controlled constant. However, genre Pop, genre R&B, genre Rock, and genre Films are negatively correlated with accuracy when other covariates are controlled constant. We have also discovered that genre In the following, we will give detailed analysis by plotting the accuracy with respect to the factors.

## Production and Analysis
We downloaded hundreds of songs from QQ Music through subscription.
We first retrieve the lyrics of our songs and format the information into a proper data frame, which is added to our s3 bucket.
Bitrate, sample rate, genre, and tempo are factors that might influence the accuracy of translation. Nowt that we get a sense of how they influence the accuracy of Amazon Transcribe's performance, we take another sample of about 600 songs to further analyze the influence of sample rate, genre, tempo, together with bitrate, on the performance of Amazon Transcribe, by showing plots.
Here is the retrieval of information of the songs.

![code](/images/code2.png)

The information of the songs in the new sample are displayed as below:

![df](/images/newtable.png)

The correlation of tempo, bitrate, sample rate, and accuracy are shown in the heatmap:

![heatmap](/images/heatmap2.png)

Doing regression of accuracy on bitrate, we get:

![accuracy_bitrate](/images/accuracy_bitrate.png)

As the regression line suggests, accuracy of Transcribe's performance increases with greater bitrate. This is intuitive, as greater bitrate usually implies better sound quailty. The negative correlation between accuracy and bitrate obtained in the previous session might be due to limitation of a small sample size.

Doing regression of accuracy on sample rate, we get:

![accuracy_samplerate](/images/accuracy_samplerate.png)

The regression line suggests a positive correlation between accuracy and sample rate. This also reinforces our notion that with higher sample rate, the audio properties are better captured by the computer.

We then plot accuracy v.s. tempo:

![accuracy_tempo](/images/accuracy_tempo.png)

As it shows, there is no remarkable relationship between accuracy and tempo.

While we cannot treat genre as a quantitative variable, we do it categorically and analyze the distribution of accuracy in each genre:

![accuracy_genre](/images/accuracy_genre.png)

It is interesting to observe that Amazon Transcribe works best with songs in rap/hip-hop genre. Is it because rap songs are less melodic? That might be the answer.

## Architecture Overview

![Architecture](/images/Architecture.png)

To summarize our project, as illustrated with the AWS architecture graph above, we collected and processed multiple audio data, stored them through S3 bucket, and used AWS Transcribe with Amazon Sagemaker as a channel to perform our analysis. Finally, outputs are passed on to final synthesis and analysis.

# Conclusion & Discussions
In this blog post, we illustrated how to use Amazon Transcribe to genereate lyrics from different songs and tested its accuracy and reliability.
As shown in the output, at a significant level of 99%, correlations between bitrate and accuracy and sample rate and accuracy are found whereas no significant correlation between tempo and accuracy is shown. For different genres, Amazon Transcribe's performed the best with Hip Pop music and it is most susceptible when applying to Rock music. The accuracy of retrieving correct lyrics depends on multiple regressors including bitrate, sample rate, genre of the song, and tempo of the song. The accuracy rate is on average is reported to be 40% on average. However, since our input data was compressed randomly, they have various quality of bitrate and sample rate. Normally, audio we obtain through online resources have relative high quality that their bitrates are typically in the range of 192k to 320 k and sample rate are higher than 44100. Thus, we believe the 75% percentile accuracy at 63.9% would be a good reference for a normal performance of Amazon Transcribe. A rate at 63.9% indicats a general reliability for retriving instant transcription, saving labor input and facilitating those who have difficulties hearing words appropriately.
There are some limitations discovered when experimenting AWS Transcribe. First of all, the sample we used is not generalized enough -- we only tested Top 100 English Songs from Billboard over 7 years. Therefore, we don't know how accurate AWS Transcribe will retrieve lyrics for songs in other languages and for songs that are not as popular as Top 100 Billboard songs. Secondly, we only examined how 4 factors (bitrate, sample rate, genre of the song, and tempo of the song) relating to accuracy and we found that song tempo would not affect the accuracy largely. There might exist other factors that could be tested and explored to improve the accuracy of AWS Transcribe while transcribing lyrics. Lastly, we also postulate that AWS Transcribe may not be originally trained for retrieving lyrics given its performance of accuracy.

# Citations
https://www.rtinsights.com/real-time-translation-machine-challenges/

https://www.dreamstime.com/photos-images/music-nerd.html

https://www.izotope.com/en/learn/digital-audio-basics-sample-rate-and-bit-depth.html

Created in Deepnote
