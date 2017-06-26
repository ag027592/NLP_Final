# NLP Final Project / 周惶振 (104061701) 何元通 (105062575)

<h1>Amplifying a Sense of Emotion toward Drama-
Long Short-Term Memory Recurrent Neural Network for dynamic emotion
recognition <br>

</h1>

<h2>Introduction</h2>
<p>We want to use the NNIME database to study the emotion behavior (such as arousal and valence state) in small duration (like in
real time), and to augment a sense of emotional feeling with visual demonstration. After all of this, we just wonder how the emotion application can be. Therefore, we think of the amplification of the emotion in video. There are a lot of video that you would feel awkward watching it because of its boring and no effect. So we want to amplify the context in the video to make the video better.
</p>

<h2>Dataset Description: NNIME-Emotion Corpus</h2>
<p>The increasing availability of large-scale emotion corpus with advancement in emotion recognition algorithms have enabled the emergence of next-generation humanmachine interfaces. The database is a result of the collaborative work between engineers and drama experts. This database includes recordings of 44 subjects engaged in spontaneous dyadic spoken interactions.
The multimodal data includes approximately 11-hour worth of audio, video, and electrocardiogram data recorded continuously and synchronously. The database is also completed with a rich set of emotion annotations of discrete and continuous-in-time annotation from a total of 50 annotators per subject.
The emotion annotation further includes a diverse perspectives: peer-report, directorreport, self-report, and observer-report. This carefully-engineered data collection and annotation processes provide an additional valuable resource to quantify and investigate various aspects of affective phenomenon and human communication. To our best knowledge, the NNIME is one of the few large-scale Chinese affective dyadic inter-action database that have been systematic-ally collected, organized, and to be publicly released to the research community.</p>

<h2>Result</h2>

|Spearman Correlation|Activation|Valence|
|---|:---:|:---:|
|SVR(Audio)|0.32|0.09|
|LSTM-RNN(Audio)|0.43|0.13|
|SVR(Text)|0.43|0.32|
|LSTM-RNN(Text)|0.1|0.04|
