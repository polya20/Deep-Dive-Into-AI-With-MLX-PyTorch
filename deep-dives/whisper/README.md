# Menny the Sonic Whisperer

![menny-avatar.png](images%2Fmenny-avatar.png)

In retrospect, I realize there was a pivotal chapter absent from Part IV of my first book, following the insightful journey of `Tenny the Vision Weaver`. 

[Chater 16 - Tenny the Convoluter](..%2F..%2Fbook%2F016-tenny-the-convoluter%2FREADME.md)

[Chapter 17 - Tenny the Vision Weaver](..%2F..%2Fbook%2F017-tenny-the-vision-weaver%2FREADME.md)

We've journeyed alongside Tenny through the realms of convolution and vision, uncovering the mysteries of how machines perceive and interpret visual stimuli. But an equally compelling question looms: how do these intelligent systems apprehend and process auditory information? How do they decode the intricate tapestry of sounds that make up our world and our spoken words? How do they extract meaning from these auditory signals, and more intriguingly, how do they understand us?

In this chapter, we embark on a fascinating exploration into the realm of auditory perception and understanding within the domain of machine learning, particularly through the lens of the MLX framework. Our focus will be on the _**Whipser**_ model, an innovative construct that exemplifies the auditory capabilities of these systems.

Before delving into the intricacies of the model, let's first sketch a broad outline of its workings. This will provide a foundational understanding that will aid in comprehending the more detailed aspects we will explore later. 

Through this exploration, we aim not only to understand the model's technicalities but also to appreciate the poetic symphony of sounds and their interpretation by machines. Menny, embodying the essence of MLX, will guide us through this acoustic labyrinth, revealing how these systems don't just 'hear' but truly 'listen' and 'understand.'

## The Whisper Model in a Nutshell: Audio Processor + Transformer + Language Model

The Whisper model represents a groundbreaking stride in the realm of auditory machine learning, particularly in the context of how machines process and understand sound. At its core, the Whisper model is a sophisticated blend of audio processing techniques and advanced neural network architectures, specifically leveraging the power of transformers and language models.

The journey of the Whisper model begins with audio processing. This initial phase involves capturing and decoding audio signals, transforming these complex waveforms into a structured format that is amenable to machine learning algorithms. This process not only cleanses the audio stream but also extracts critical features that are essential for the subsequent stages.

Once the audio has been processed, the Whisper model pivots to its core components: a transformer and a language model. The transformer, renowned for its effectiveness in handling sequential data, delves into the intricacies of sound patterns, extracting contextual information and nuances. It's in this phase that the model truly begins to 'understand' the audio, discerning patterns and structures that go beyond mere sound waves.

Parallelly, the language model component comes into play, bringing a deeper understanding of linguistic elements and semantics. This integration is vital, as it allows the Whisper model to not only recognize sounds but also interpret them, making sense of language and meaning within the audio. It's here that the model bridges the gap between mere sound recognition and true auditory comprehension.

In essence, the Whisper model is not just about processing sound; it's about endowing machines with the ability to comprehend and interact with the auditory world in a manner akin to human understanding. It stands as a testament to the advances in machine learning and its capability to imbue machines with a near-human level of auditory perception.

But first, to truly grasp the Whisper model, we must start at the beginning: audio processing. So, let's dive into that. What exactly is audio processing, and how does it function?

For those who are new to the world of audio processing, I strongly recommend taking a moment to read the sidebar on _Random Variables and Probability Distributions_. This will give you a solid foundation in the key concepts that underpin our exploration in this deep dive.

[A-Primer-On-Random-Variables-And-Probability-Distributions.md](..%2F..%2Fbook%2Fsidebars%2Fa-primer-on-random-variables-and-probability-distributions%2FA-Primer-On-Random-Variables-And-Probability-Distributions.md)

You might be asking, "What does audio have to do with random variables and probability distributions?" The connection is quite straightforward: digital audio involves sampling from real-world, analog audio.

![setup-music-studio.png](images%2Fsetup-music-studio.png)

_Some images are intentionally blurred and edited to protect privacy._

Lucky for you, I'm not only an audiophile but also deeply passionate about everything related to audio and video. 

![setup-ht.png](images%2Fsetup-ht.png)

In my explorations, I've stumbled upon something extraordinary within the realm of audio - a discovery that almost feels like uncovering one of the universe's great secrets. 

![setup-music-studio2.png](images%2Fsetup-music-studio2.png)

And this revelation is fundamentally mathematical, suggesting that the intricacies of the universe, perhaps crafted by some higher power, are deeply rooted in mathematical principles. 

This might sound far-fetched, and believe me, I'm neither a religious nor a spiritual individual. My faith lies in the power and precision of mathematics. So, if you're curious to discover this intriguing piece of evidence that I've found in the patterns and structures of audio, keep reading. The journey into the mathematical heart of audio is not just fascinating; it might change how you perceive the world around us.

And here's an encouraging thought: the math behind this secret isn't overly complex. In fact, it's quite straightforward. Yet, the implications of this simple math are nothing short of profound.

Analog audio is a continuous signal, while digital audio is a discrete representation of this signal. To fully understand this, it's crucial to familiarize yourself with the fundamental differences between analog and discrete data.

## Analog vs. Discrete Data - The Essence of Sampling Rate in Audio Processing

In the realm of audio processing, particularly when we delve into models like Whisper, understanding the distinction between analog and discrete data is pivotal. This distinction forms the foundation of how audio is captured, processed, and interpreted by digital systems.

### Analog Audio: The Continuous Symphony

Analog audio can be visualized as a continuous wave, a smooth, flowing representation of sound. This is the form in which sound exists naturally in our environment – as pressure waves traveling through the air. The key characteristic of analog audio is its continuity; it's an unbroken signal with an infinite resolution. 

Imagine speaking or playing a violin. The sound produced is a seamless wave, fluctuating in amplitude and frequency over time. Analog recording methods, like vinyl records, capture this continuous wave directly, preserving the rich, detailed nuances of the sound.

### Discrete Data: The Digital Interpretation

On the other side, we have discrete data, which is the backbone of digital audio. Discrete, in this context, means separate and distinct. Digital audio is not a continuous wave but a series of individual samples taken from the analog signal at specific intervals. Each sample is a snapshot of the audio wave at a particular moment, quantized into a digital value that represents the amplitude of the wave at that instant.

The process of converting analog audio into digital form is called 'sampling'. It involves measuring the amplitude of the analog wave at regular intervals, known as the sampling rate, and then converting these measurements into digital values. The most common example is the Compact Disc (CD) audio, which uses a sampling rate of 44.1 kHz – meaning it takes 44,100 samples of the audio wave every second.

The _Nyquist Theorem_, a fundamental principle in the field of digital signal processing, states that in order to accurately capture a continuous signal (like an audio signal) without losing information, it must be sampled at least at twice the highest frequency present in the signal. This minimum rate is known as the Nyquist rate. For example, since the human audible range extends up to 20 kHz, audio for human consumption is typically sampled at 44.1 kHz, which is slightly more than twice the highest frequency we can hear. This theorem is crucial because it provides the theoretical foundation for converting continuous analog signals into discrete digital signals without losing critical information, ensuring that the digital representation closely mirrors the original analog signal.

Indeed, it's all about object-oriented learning, where precision and quantization come into play. This approach methodically breaks down complex concepts like digital signal processing into more manageable, object-based components, allowing for a clearer understanding and more precise application of these principles.

[Precision-And-Quantization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fprecision-and-quantization-made-easy%2FPrecision-And-Quantization-Made-Easy.md)

### The Implications for Audio Processing

Understanding the transition from analog to discrete data is crucial for audio processing in machine learning models like Whisper. The quality of digital audio depends heavily on two factors: the sampling rate and the bit depth (the number of bits used to represent each sample). Higher sampling rates and greater bit depths can more accurately represent the original analog wave, leading to higher-quality digital audio.

However, this increased fidelity comes at a cost: larger file sizes and more processing power required to handle the data. In machine learning, especially in models dealing with large datasets or real-time processing, these factors become critical considerations. The art of audio processing in this context lies in striking the right balance – maintaining audio quality while optimizing for computational efficiency.

In summary, the journey from the continuous world of analog audio to the discrete realm of digital audio is not just a technical transition but a fundamental step in preparing audio for the sophisticated processes of machine learning models. Understanding this journey is key to unlocking the deeper functionalities and capabilities of models like the Whisper.

### Video Equivalent of Audio Sampling - Frames Per Second, Resolution, and Bit Depth

In the digital video domain, understanding the concepts of frames per second (fps), resolution, and bit depth is as crucial as comprehending sampling in audio processing. Digital videos are composed of a series of frames, which are essentially individual still images. When these frames are displayed rapidly in succession, they create the illusion of continuous motion, much like how discrete samples in digital audio form a continuous sound when played sequentially. The resolution of a video refers to the amount of detail in each frame, measured in pixels, while bit depth determines the color depth of each pixel. Together, these aspects are critical in how digital video replicates the fluidity of real-world motion.

The natural motion we perceive in real life, however, does not operate on a 'frames per second' basis. Our real-world experience of motion is a continuous flow, distinctly different from the discrete, segmented nature of digital video. Frames per second, resolution, and bit depth in digital video are constructs designed to emulate the uninterrupted, seamless motion we see in the physical world. Understanding these digital video fundamentals is key when extending our approach from audio to a broader, object-oriented perspective in multimedia learning and processing. This knowledge allows for a more nuanced integration of audio and video data, bridging the gap between the segmented nature of digital media and the continuous flow of real-world experiences. This enhanced understanding is invaluable in developing more sophisticated and realistic multimedia learning algorithms and applications.

### We're Compressing the World After All

In essence, our task in digital processing is to compress the rich, continuous analog data of real-life experiences into discrete digital form. This compression is not just a technical necessity but an integral part of the digital revolution. The key challenges we face are efficiency, fidelity, and balance. How do we compress data most effectively? How do we minimize the inevitable loss of information during this conversion? And perhaps most crucially, how do we strike the perfect balance between maintaining high fidelity to the original analog source and ensuring the computational efficiency required for modern applications?

It's essential to understand that terms like 'lossless', 'high-fidelity', 'high-definition', and even '8K videos' are all descriptors of digital data that has been compressed. These terms often imply a level of quality or closeness to the original source, but it's crucial to remember that all digital data is inherently discrete and, thus, represents a departure from its analog origin. The term 'lossless' within the digital domain means that there's no additional signal loss during the process of digital encoding or compression. However, the moment we convert analog data into digital form, a certain degree of information loss is unavoidable. 

In the journey from analog to digital, every choice we make in terms of sampling rate, bit depth, and compression algorithms impacts the final output. While we can get remarkably close to the original with advanced technology, a 100% lossless digital representation of analog data remains an elusive goal. Thus, our focus should be on optimizing these processes, understanding the trade-offs involved, and continually pushing the boundaries of technology to minimize this gap, all while keeping in mind the limitations inherent in the digital representation of our analog world.

### The Reality Check: Our Brain as a Selective Digital Processor

It might come as a surprise, but in a way, our brain functions somewhat like a digital processor, especially when it comes to handling sensory data. Don't be misled into thinking that our brain processes information in a purely analog manner. While it's incredibly complex and capable, it isn't equipped to handle the full range of analog data, whether that's audio, visual, or any other sensory input.

Normalization. Sound familiar? Indeed, our brain comes equipped with numerous natural biological mechanisms for normalization. 

[Normalization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fnormalization-made-easy%2FNormalization-Made-Easy.md)

To illustrate this point, let's focus on audio, though you can extrapolate this concept to other sensory inputs from an object-oriented perspective. 

![audible-spectrum.png](images%2Faudible-spectrum.png)

Humans are capable of hearing only a specific subset of sound wave frequencies. This range, typically from 20 Hz to 20 kHz, is known as the audible spectrum. It's a limited segment of the broader spectrum of sound waves that exist in our environment. 

Consider other species: dolphins and bats, for instance, can perceive ultrasonic frequencies that fall well outside the human audible range. These ultrasonic frequencies are crucial for their navigation and communication.

As we age, our ability to perceive certain frequencies diminishes. This is why high-pitched sounds often become harder to hear for older individuals. Ingeniously, young people sometimes exploit this by using high-frequency sounds to communicate, creating a sort of 'secret language' that adults can't decipher.

So why is our brain wired this way? It's a matter of evolutionary adaptation. The human brain hasn't evolved to process the entirety of sound wave frequencies. Instead, it's fine-tuned to a range that's been most relevant for our survival and communication needs. 

This selective processing is akin to the way digital systems handle data. Just as our brain filters and processes a specific range of frequencies, digital audio systems are designed to sample and encode a selected range of sound waves. Both systems, in their own ways, simplify and compress the vast complexity of the world into more manageable, efficient forms. This understanding not only underscores the limitations of our perception but also highlights the parallels between the way our brains and digital systems process information, albeit with different mechanisms and capabilities.

### Fun Math Fact: The Arithmetic of Tuning in Music

Here's the intriguing secret I promised: at the heart of tuning musical instruments lies a fundamental mathematical operation - simple arithmetic.

As a musician who plays various instruments like the keyboard, guitar, bass, and drums, I've often engaged in the tuning process. Most of these instruments, especially the non-digital ones, require precise tuning, which, at its core, is a mathematical exercise.

Consider orchestras, which typically tune to a specific frequency, commonly A4 = 440 Hz, known as the _concert pitch_. Tuning an instrument involves adjusting its strings or other components to align with this standard frequency.

Let's explore the standard tuning of a guitar. Each string is tuned to a specific pitch, and it's essential for creating harmonious chords and melodies. Here's a breakdown of the standard tuning for a six-string guitar:

1. **6th String (Low E)**: Tuned to **E4**, which is an octave lower than **E5**.
2. **5th String (A)**: Set to **A4**.
3. **4th String (D)**: Tuned to **D4**, which is a perfect 5th below **A4**.
4. **3rd String (G)**: Aligned with **G4**, a perfect 4th below **A4**.
5. **2nd String (B)**: Set to **B4**, which is a major 3rd above **A4**.
6. **1st String (High E)**: Tuned to **E5**, an octave higher than **E4**.

Remember, these standard tunings serve as a foundation for playing various chords, scales, and songs on the guitar.

These tuning intervals reveal a fascinating arithmetic relationship centered around the A4 = 440 Hz pitch. For instance, what would be the frequency of A3, the perfect lower octave of A4? By applying simple division twice, as each octave represents a halving of frequency, 440 Hz divided by 2 gives us 220 Hz for A3, and dividing 220 Hz by 2 once more gives us 110 Hz for A2. This arithmetic foundation in music tuning is not just a technical detail but a beautiful illustration of how math and art intertwine in our everyday experiences.

Do you still believe this is merely a coincidence?

In physics, the laws of thermodynamics suggest a progression from order to disorder over time. Take the law of entropy, for example: it posits that the entropy, or the measure of disorder, in an isolated system never decreases. This law helps to explain why things deteriorate or become more chaotic as time progresses. It's like the common saying: you can't unscramble an egg.

But what about audio that seems as jumbled as an omelet? Can we "unscramble" it? The answer lies in mathematics. You might assume that a distorted recording of your voice is beyond repair, but mathematically, it's often possible to reverse the process. This means that even if you mumble or scramble your voice, it's not necessarily secure from mathematical analysis and reconstruction. Surprisingly, there's often a way to unscramble audio and make it intelligible again, thanks to the power of mathematical algorithms.

All those audio tools that restore distorted audio and eliminate noises operate on mathematical algorithms. These sophisticated algorithms are meticulously crafted to dissect the audio, sifting through the distortion to reconstruct it into a clearer, more comprehensible form. This process mirrors how our brains process auditory information, instinctively filtering out ambient noise to concentrate on important sounds.

![noise-canceling.png](images%2Fnoise-canceling.png)

Consider noise-canceling headphones as a practical example of applied mathematics. They utilize a simple yet effective mathematical principle: by inverting the phase of background noise and overlaying it onto the original sound, the noise is effectively neutralized.

You can test this concept with your favorite Digital Audio Workstation (DAW). Record a sound, create a duplicate track, reverse the phase of the duplicate, and play them simultaneously. You will observe that the sound seems to disappear - a phenomenon known as phase cancellation. This is the fundamental principle behind the technology in noise-canceling headphones.

## Fourier Transform: Unraveling the Complexities of Sound

To delve deeper into the intricacies of audio processing, envision the auditory world as an intricate tapestry of sound waves, each distinguished by its unique frequency and amplitude. The challenge lies in deciphering this elaborate, multidimensional soundscape: how do we distill essential information from this symphony of sounds?

The answer is encapsulated in one word: layers. The secret to navigating through this complex auditory landscape is by deconstructing it into its fundamental components. Enter the Fourier Transform, a powerful mathematical tool that serves as the cornerstone of this deconstruction.

Imagine you're working with a layered Photoshop image. Just as you can isolate and modify individual layers in the image, the Fourier Transform enables us to "peel off" and examine the individual frequencies within a sound wave. This process is akin to breaking down a complex melody into its individual notes.

Named after the French mathematician Joseph Fourier, who formulated this transformative concept in the early 19th century, the Fourier Transform revolutionized how we understand and manipulate sound. It's not just a mathematical technique; it's a lens through which we can view and interpret the diverse frequency components that make up a sound. Whether it's isolating a specific instrument in a symphony or cleaning up a noisy audio recording, the Fourier Transform provides the means to dissect sound into its purest elements, offering unparalleled insight into the very essence of audio processing.

### Expanding the Concept from the Object Oriented Perspective - Blade Runner and Ray Tracing

![blade-runner-midjourney.png](images%2Fblade-runner-midjourney.png)

Indeed, I'm a great admirer of Ridley Scott's work, particularly the original "Blade Runner." If you haven't seen it, I highly recommend it, not just for its cinematic brilliance but also for a key scene that beautifully illustrates a concept in computer graphics – ray tracing. In the film, protagonist Deckard analyzes a photograph, zooming in and enhancing it to reveal previously hidden details. While the scene might have seemed like futuristic fantasy at the time, it's a perfect metaphor for the concept of ray tracing, a technology we understand and utilize today.

Ray tracing, unknown at the time of the film's release, has since become a cornerstone in the field of computer graphics. It's a method that simulates the behavior of light, tracing the paths of light rays as they interact with objects in a scene. This simulation creates highly realistic images by mimicking the natural behavior of light in the real world.

Isn't it fascinating? "Let there be light," and not only can we observe it, but we can also mathematically trace it. This brings us to an awe-inspiring realization: our universe is essentially a mathematical construct. With mathematics, we can reconstruct what we see, hear, and feel.

It's truly incredible when you think about it. The secrets of the universe that I promised are indeed rooted in mathematics. Regardless of where you are, whether delving into classical or quantum physics, the underlying constant is numbers, the language of math. It's this universal language that enables us to understand and describe the intricacies of the world around us.

Okay, now we're ready to move on. 

### Getting the Sense of Simple Audio File

Let's start with a simple audio file. The `hello.wav` contains some words spoken with AI voice at ElevenLabs.

```python
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read the WAV file
file_path = './data/hello.wav'
sample_rate, data = wavfile.read(file_path)

# Generate time axis
time = [float(n) / sample_rate for n in range(len(data))]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time, data)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

```

Keep in mind that the `wave` module in Python's standard library is designed to handle only uncompressed PCM audio. If you encounter an error message like "unknown format: 3" while using this module, it's likely because the WAV file you're trying to read is in a compressed or non-PCM format. This specific error indicates that your file is possibly in a floating-point format, which corresponds to the format code 3. To manage such scenarios, it's advisable to utilize a more versatile audio processing library such as `scipy`, which is equipped to handle a broader spectrum of audio formats, including those that the `wave` module might not support.

![wave-inspection.png](images%2Fwave-inspection.png)

The code you've run reads an audio file and visualizes its waveform using Python. Here's a breakdown of what the code does:

1. **Reads the Audio File**: The code uses a Python library to read the `.wav` file, extracting the sample rate (number of samples per second) and the audio data itself (the amplitude of the sound wave at each sample point).

2. **Generates Time Axis**: It then calculates a time axis to correspond with the audio samples, which is necessary for plotting the waveform. The time for each sample is computed by dividing the sample index by the sample rate, giving the time in seconds for each point in the audio data.

3. **Plots the Waveform**: Using `matplotlib`, the code plots the audio data against the time axis. This plot is the waveform of the audio file, showing how the amplitude of the sound wave varies over time.

The resulting image is a visualization of the waveform of the `hello.wav` audio file. You can see the variations in amplitude over time, which correspond to the different sounds (phonemes and words) spoken in the audio file. The pronounced peaks and troughs represent louder and softer sounds, respectively. 

In the context of the audio data:

- **Amplitude**: The vertical axis represents the amplitude of the audio signal. Amplitude corresponds to the volume or loudness; larger amplitudes mean louder sounds.

- **Time**: The horizontal axis shows time in seconds. This indicates when each sound occurs in the audio file.

The waveform gives you a visual representation of all the sounds in the file. For example, you might see a pattern repeating for certain sounds or notice where there are pauses in speech (areas where the waveform is close to zero).

By analyzing the waveform, you can infer certain properties of the audio file, such as the rhythm of speech, the emphasis on certain words, or where different words or sounds begin and end. This kind of visualization is fundamental in audio editing, processing, and analysis.

We need to dig a bit deeper to understand the intricacies of audio processing. Let's take a closer look at some of the terms you will encounter in this domain.

## The Anatomy of Audio

![adobe-audition.png](images%2Fadobe-audition.png)

These terms are all critical concepts in the field of digital audio processing, and understanding them is key to manipulating and analyzing sound effectively.

### Resolution of the Audio

The resolution of audio refers to the detail or precision with which it represents the original analog signal. It is determined by two primary factors: bit depth and sampling rate. Higher resolution audio has greater fidelity to the original sound.

### Sampling Rate

The sampling rate is the number of samples of audio carried per second, measured in Hz or kHz. It defines the temporal resolution of audio, meaning how many times per second the audio waveform is measured during the analog-to-digital conversion process. A common standard for music is 44.1 kHz, which means the audio is sampled 44,100 times per second. According to the Nyquist Theorem, to capture all frequencies up to 20 kHz (the upper limit of human hearing), you need a sampling rate of at least twice that frequency. Recall the Niqyist Theorem mentioned earlier.

### Bit Depth

Bit depth refers to the number of bits of information in each sample. It determines the audio signal's amplitude resolution. The higher the bit depth, the more detailed the amplitude of each sample, which contributes to the overall dynamic range and can reduce the noise floor. Common bit depths are 16-bit, 24-bit, and 32-bit.

### Dynamic Range

Dynamic range is the difference between the quietest and loudest volume of an audio track that can be recorded or reproduced without distortion. It's essentially the range of volume that a system can handle and is closely related to bit depth. The higher the bit depth, the greater the potential dynamic range.

### Headroom

Headroom in audio refers to the amount of space or "buffer" between the peak level of your audio signal and the maximum level that your audio system can handle before distortion occurs. It's a safety margin to prevent clipping (distortion of a signal by its being "cut off" at the maximum capacity of the system).

### Noise Floor

The noise floor is the measure of the signal noise level in an audio recording. It represents the lowest level of the analog or digital system's noise, below which the signal is not discernible. In a high-quality recording or sound system, you want the noise floor to be as low as possible to ensure the clarity and cleanliness of the audio.

In our context, when dealing with audio processing and analysis, it's crucial to select appropriate sampling rates and bit depths to capture the full range of the sound without introducing noise or distortion. This ensures that the final processed audio maintains the nuances of the original sound, which is particularly important in professional audio environments or any application where audio quality is paramount.

### Phase

In the context of audio and signal processing, "phase" refers to the position of a point in time on a waveform cycle. A waveform cycle is the complete motion of a signal from its starting point, through its maximum amplitude, back down to the minimum amplitude, and returning to the starting point.

More technically, phase is the fraction of the wave cycle that has elapsed relative to the origin. It's usually measured in degrees, where 360 degrees is one full cycle, or in radians, where `2π` radians equals one full cycle.

Phase becomes a particularly important concept when dealing with multiple waves at the same frequency. If two waves are "in phase," their peaks and troughs match up with each other in time. If they are "out of phase," the peaks of one wave correspond to the troughs of another, and they can cancel each other out to some degree when combined, a phenomenon known as phase cancellation. As we've seen, this is the principle used in noise-canceling headphones to reduce unwanted ambient sounds.

In sound waves, phase differences can cause constructive interference (when waves add together to make a larger wave) or destructive interference (when waves cancel each other out). This is most easily heard in the form of phase cancellation when two identical sounds with a phase difference are played together, resulting in a reduced or muted sound.

## Video Equivalent - Resolution, Dynamic Range, Bit Depth, and Color Space

In the visual realm, concepts akin to audio processing terms also play pivotal roles, shaping the quality and realism of video content. Given their polymorphic natures, these concepts transfer seamlessly between audio and video, with each term having a visual counterpart that is integral to video production and analysis.

### Resolution

In video, resolution refers to the number of pixels that compose the image on the screen, typically represented by the width and height of the display or image (e.g., 1920x1080). Higher resolution means more pixels and, consequently, more detail that can be seen in the image, contributing to the clarity and sharpness.

### Dynamic Range

Just as with audio, dynamic range in video describes the range between the darkest and brightest parts of the image. In video, this is often referred to as contrast ratio. A greater dynamic range allows for a picture that can display deeper blacks and brighter whites, revealing more detail in the shadows and highlights, and delivering a more life-like image.

You love HDR videos and games, don't you?

_HDR_ stands for _High Dynamic Range_. In imaging and photography, HDR is a technique that captures, processes, and reproduces content in such a way that the detail of both the shadows and highlights of a scene are preserved and clearly visible. Traditional imaging and display techniques may lose detail in the darkest and brightest areas of a picture due to the limited dynamic range – the contrast between the darkest and lightest elements. HDR expands this range significantly.

In the context of video and digital displays, HDR content is produced by capturing and combining multiple photographs of the same subject at different exposure levels. For video displays, HDR technology allows screens to show a more vivid and perceivable range of colors and luminance. This results in images that can have bright, glowing highlights and deep, inky shadows, closely mimicking the range of light intensities found in the real world.

HDR technology in video and photography requires:

- **HDR-Compatible Cameras**: To capture a wider range of luminance levels than is possible with standard digital imaging techniques.
- **HDR-Compatible Displays**: To show the greater range of luminance levels. These displays are capable of producing higher brightness and contrast levels than standard dynamic range (SDR) displays.
- **Specialized Software**: To combine images taken at different exposures, or to process the 'wider' data captured by HDR cameras for viewing.

HDR is particularly important in modern content consumption, as it brings a more lifelike and immersive visual experience to viewers, especially when combined with higher resolutions and wider color spaces in newer UHD (Ultra High Definition) displays.

Thus, if you encounter a video or game that supports HDR yet the imagery appears dull or washed-out, it's likely due to the viewing platform or display lacking HDR capability. For the full visual impact of HDR content, both the device you're using to view the content and the display itself must be equipped with HDR support. Only with the proper HDR-compatible hardware can you experience the intended richness of contrast and color that HDR provides.

### Bit Depth

In video, bit depth is related to color depth. It represents the number of bits used to indicate the color of a single pixel. The more bits, the more colors that can be represented, allowing for smoother gradients and less banding. For instance, a bit depth of 8 bits per color channel can display 256 shades of that color, while a bit depth of 10 bits can display 1024 shades.

### Color Space

Color space defines the range of colors, or color gamut, that a video can represent. Different color spaces capture varying levels of color detail and are suited for different purposes. For example, sRGB is the standard color space for the internet, while Adobe RGB has a wider gamut suitable for professional printing. In video, common color spaces include Rec. 709 for HD video and Rec. 2020 for 4K and HDR video, with Rec. 2020 offering a much wider range of colors than Rec. 709.

Each of these terms is essential in the craft of video production and post-production. They ensure that the end result is as true to life as possible or as creatively envisioned. When optimized, they work together to create a rich, immersive visual experience that can convey a broad spectrum of emotions and narratives, much like how the interplay of frequency, amplitude, and timbre can create a moving auditory experience.

Alright, I might have let my enthusiasm for the subject carry me away. It's easy to get engrossed in drawing parallels between audio and video, as they share many underlying principles. However, the crux of the matter is the object-oriented approach that's essential for comprehending the complex tapestry of our sensory experiences. 

_The Zen of Smart Effort_
[The-Zen-Of-Smart-Effort.md](..%2F..%2Fessays%2Flife%2FThe-Zen-Of-Smart-Effort.md)

With this mindset, we can dissect and appreciate the nuances of both audio and visual worlds. Now, let's circle back to the realm of audio.

## A Bit More Advanced Concepts

The concepts we've discussed provide a solid foundation for general audio processing. But for those looking to delve deeper into the specialized field of speech recognition, a deeper understanding of advanced techniques and principles is essential. Let's delve into these more complex ideas to expand our knowledge in the realm of speech processing.

### How Sound Works

![newtons-cradle.png](images%2Fnewtons-cradle.png)

Air molecules oscillate back and forth at various frequencies but don't travel with the sound wave. When these oscillations reach your ear, they are interpreted as sound. Consider the analogy of Newton's cradle: when the first ball strikes the second, the impact is transmitted through the intermediate balls, causing only the last ball to swing outward. Similarly, sound waves propagate through the medium of air by transferring energy from one molecule to the next, while the individual molecules themselves remain in the same average position. This is akin to how a wave travels through a stadium crowd—people stand up and sit down in sequence, creating a visible wave around the stands, yet each person remains in their spot.

It's a common misconception that air molecules themselves travel over long distances when sound is produced. In reality, while these molecules do vibrate and transfer energy when sound waves pass through, they largely stay in their original positions. The molecules oscillate around a fixed point, creating waves of compression and rarefaction that move through the air, allowing sound to propagate without the molecules themselves traveling from the source to your ear.

The speakers in your headphones or audio system function by vibrating back and forth, pushing against the air to create pressure waves. These pressure waves ripple outward, ultimately reaching your ear, where they are detected and decoded as sound by your auditory system. The pitch of the sound you hear corresponds to the frequency of these waves—the number of times the air pressure peaks in a second—while the loudness or volume of the sound is a result of the waves' amplitude, or the height of those pressure peaks. This is the fundamental process behind the perception of sound.

Recording audio with a microphone is essentially the reverse process of playing it through speakers: it involves converting sound waves into electrical signals. The diaphragm of a microphone moves in sync with the incoming pressure waves of sound. These mechanical vibrations are then transformed into corresponding electrical signals. This translation from physical sound waves to electrical signals is the crux of analog-to-digital conversion, laying the groundwork for storing and manipulating audio in the digital realm.

### Digital to Analog, Analog to Digital Conversions

In the world of audio technology, DACs (Digital-to-Analog Converters) and ADCs (Analog-to-Digital Converters) play pivotal roles in bridging the gap between the digital and analog realms. These conversions are not just technical necessities; they're the alchemy that transforms the zeros and ones of digital files into the rich, textured experiences of sound and music that resonate with us—and vice versa. Let's delve deeper into these transformative processes.

#### Analog to Digital Conversion (ADC)

When sound waves are captured—say, through a microphone—the resulting analog signals represent continuous waves of varying voltages that correspond to the sound's pressure waves. An ADC takes these analog signals and translates them into a digital format. How does it accomplish this? By sampling the signal at regular intervals (the sampling rate) and measuring the amplitude of the wave at each point (quantization). 

The precision of this process is determined by two key factors: the bit depth, which affects the granularity of the amplitude measurements, and the sampling rate, which affects the temporal resolution of the digital representation. A higher bit depth allows for a more precise measurement of the sound wave's amplitude, leading to a finer gradation of sound levels and a lower noise floor. A higher sampling rate captures more of the sound wave's nuances, ensuring that higher frequencies are accurately represented and preventing aliasing, a kind of distortion that can occur when the sampling rate is too low.

#### Digital to Analog Conversion (DAC)

![dave1.png](images%2Fdave1.png)

On the other side of the spectrum, a DAC performs the reverse operation. It takes digital audio files, which are essentially long sequences of numbers, and converts them back into analog signals. These analog signals can then be amplified and sent to speakers or headphones to produce sound that we can hear. The DAC must accurately reconstruct the original analog wave from the discrete digital samples, a process that involves interpolation to fill in the gaps between samples.

The quality of a DAC is critical in determining the fidelity of the playback. Higher quality DACs are better at minimizing jitter (tiny timing errors in the conversion process), handling different sampling rates, and producing a clear, dynamic range of audio. In essence, a good DAC can make a digital recording sound 'analog'—warm, full, and natural.

In conclusion, ADCs and DACs are the unsung heroes of our digital audio experience. They work quietly behind the scenes, but their impact on audio quality is profound. Whether we're recording a live concert or enjoying our favorite album on a smartphone, these conversions via ADCs and DACs ensure that we can seamlessly capture and enjoy audio in a digital world, preserving the depth and detail of the original sounds.

![dave2.png](images%2Fdave2.png)

_Back in the day, I used to do unboxings and reviews._

Inside every smartphone, be it an iPhone or an Android device, you'll find both DACs and ADCs integral to the device's audio system. These tiny components wield a significant influence over the quality of sound your phone can produce and capture. For the average user, the built-in DAC and ADC may suffice, providing a sound that's perfectly acceptable for everyday listening. However, audiophiles, with their keen ears and quest for acoustic perfection, often seek out much more.

To the audiophile, the nuanced layers of sound and the depth of the audio scene are critical, and the quality of a DAC can make or break their listening experience. This discerning ear can detect subtleties that might be lost on casual listeners, which is why high-fidelity audio enthusiasts are willing to invest heavily in their setups. A top-tier DAC will come with a price tag to match, reflecting the precision engineering required to minimize noise, distortion, and coloration of the sound.

For those who prioritize audio quality, the internal DACs and ADCs of phones won't hold a candle to dedicated high-end audio equipment. These enthusiasts know that the pathway to auditory bliss is paved with premium components, where every link in the audio chain, especially a good DAC, contributes to the richness and purity of the sound.

![mscaler.png](images%2Fmscaler.png)

Audio enthusiasts are known to sometimes take their passion to the extreme, venturing into the contentious realm of upscaling audio samples, a topic that's a tale in itself for another time.

### The Mel Scale: A Human-Centric Approach to Audio Processing

In our ongoing exploration of audio processing, it's essential to understand how the human ear perceives sound and how we can model that perception in digital systems. This is where the _Mel scale_ enters the picture, serving as a cornerstone concept in the field of audio signal processing.

The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another. The scale is based on the human ear's natural response to different frequencies. Not all frequencies are perceived equally by the human ear; we are more sensitive to changes in frequency at lower frequencies than at higher ones. The Mel scale reflects this by spacing the perceived pitches in a way that is more linear to human hearing, especially in the critical speech frequency range.

The Mel scale's significance stems from its human-centric design. When processing speech and music, what matters most is how sound is perceived by humans, not how it's mathematically structured in terms of frequency. By converting the frequency domain representation of an audio signal into the Mel scale, we create a representation that more closely aligns with human auditory perception. This is particularly useful for tasks such as speech recognition, where the goal is to interpret audio as a human listener would.

In a nutshell, we're normalizing the frequency axis to align with human hearing. This is a crucial step in audio processing, as it enables us to model the human perception of sound, which is essential for developing sophisticated audio analysis and interpretation models.

[Normalization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fnormalization-made-easy%2FNormalization-Made-Easy.md)

_Mel filterbanks_ are used to convert the spectrum obtained from a Fourier transform, which is linear in frequency, into the Mel spectrum. These filterbanks are a collection of filters, each designed to pass a certain portion of the spectrum and mimic the Mel scale's warping of the frequency axis. When the Fourier transform's output passes through these filters, the result is a Mel-frequency spectrogram.

The Mel-frequency spectrogram is a two-dimensional representation of the sound: time on one axis and Mel frequency bands on the other. The intensity of each point in the Mel spectrogram represents the energy of the sound within a specific Mel frequency band at a specific time. This representation is excellent for feeding into machine learning models, as it encapsulates the nuanced features of sound in a way that's attuned to human hearing.

In summary, the Mel scale and Mel filterbanks are vital tools in audio processing that enable us to transform audio signals into a form that reflects human auditory characteristics. By doing so, we bridge the gap between the quantitative nature of digital signal processing and the qualitative aspects of human sensory experience. This synergy is crucial for developing sophisticated audio analysis and interpretation models, such as those used in the Whisper model, which aim to understand and process audio with a level of precision and nuance akin to the human ear.

### The Role of Window Functions in Signal Processing: Understanding the Hanning Window

In the realm of digital signal processing, particularly when working with audio data, window functions play a crucial role. These functions, such as the _Hanning window_ (also known as the _Hann window_), are essential in managing a common issue known as edge effects. Let's delve into the importance and application of the Hanning window in signal processing.

The Hanning window is a type of window function used to modify a signal before applying a Fourier transform. Named after Julius von Hann, an Austrian meteorologist, it's characterized by a sinusoidal shape that tapers off at the ends. This tapering is crucial; it reduces the abruptness at the edges of each frame of audio data.

When performing a Fourier transform on a segment of audio data, we typically need to isolate a specific portion or 'frame' of the signal. However, this isolation can create artificial discontinuities at the edges of the frame because the signal at the start and end of the frame doesn't necessarily match up. These discontinuities can introduce artifacts in the Fourier transform, known as spectral leakage, which distort the frequency spectrum of the signal.

The Hanning window addresses this issue by gently tapering the signal to zero at the frame's boundaries, ensuring a smooth transition. When the window is applied, it effectively 'fades in' and 'fades out' the signal at the edges, reducing the abruptness and thus minimizing spectral leakage. This smoother transition at the boundaries makes the Hanning window particularly effective for analyzing signals where continuity is vital, such as in audio and speech processing.

To understand this better, consider the analogy of using a blending brush in Photoshop. Just as a blending brush smoothly merges colors at the edges to avoid harsh lines and create a seamless transition, window functions like the Hanning window work to smoothly 'blend' the edges of an audio frame. This blending minimizes abrupt changes at the frame boundaries, thereby reducing spectral leakage and ensuring a more faithful representation of the signal’s frequency content.

In practice, applying a Hanning window to a frame of audio data involves element-wise multiplication of the window with the signal. This process modulates the amplitude of the signal, preserving the central part of the frame while attenuating the beginning and end. The result is a signal segment that aligns more naturally with the assumptions of the Fourier transform, leading to a more accurate representation of the frequency content.

In summary, the Hanning window and similar window functions are indispensable tools in signal processing. They enable more precise and reliable frequency analysis by mitigating the effects of framing on a continuous signal, thus playing a critical role in various applications, from audio analysis to communication systems.

### Understanding the Log-Mel Spectrogram in Audio Processing

The _log-Mel spectrogram_ is an enhancement of the Mel spectrogram. While the Mel spectrogram uses the Mel scale to better represent how humans perceive sound, the log-Mel spectrogram goes a step further by applying a logarithmic scale to the Mel spectrogram's amplitude. 

The logarithmic transformation is applied to the amplitude values of the Mel spectrogram. This step is crucial because our human auditory system perceives sound intensity on a logarithmic scale, not a linear one. In simpler terms, we perceive sound in a way that a doubling of the actual sound energy doesn't necessarily sound twice as loud to our ears. The logarithmic scale in the log-Mel spectrogram mimics this aspect of human hearing, making the representation of sound in this format more natural and intuitive to how we actually experience audio.

The log-Mel spectrogram, with its closer mimicry of human hearing, is especially useful in machine learning models dealing with audio data. In tasks like speech recognition, sound classification, and audio event detection, it provides a more meaningful representation of sound for algorithms to analyze. This representation allows models to focus on the aspects of sound most relevant to human listeners, leading to more accurate and effective processing and interpretation of audio data.

In essence, the log-Mel spectrogram isn't just a technical transformation—it's a bridge that connects the raw, objective measurements of sound with the subjective way we experience it, paving the way for more sophisticated and human-centric audio processing technologies.

Once more, when encountering a logarithmic scale, think of it as a form of normalization.

### Understanding Decibels: The Logarithmic Scale of Sound

In the realm of audio and acoustics, the decibel (dB) is a key unit of measurement. It exemplifies the concept of logarithmic scaling, which is essential for understanding and measuring sound in a way that aligns with human auditory perception. Let's delve into the concept of decibels and how they apply the logarithmic principle for effective normalization of sound levels.

#### The Decibel: A Logarithmic Unit

A decibel is a logarithmic unit used to express the ratio of two values of a physical quantity, often power or intensity. In the context of sound, it's used to measure sound pressure level (SPL), which is a logarithmic measure of the effective pressure of a sound relative to a reference value.

#### Why Logarithmic?

The human ear perceives sound intensity logarithmically rather than linearly. This means that when sound intensity doubles, it doesn't necessarily sound twice as loud to our ears. A logarithmic scale, like that of decibels, reflects this perception more accurately than a linear scale. 

For instance, an increase of 10 dB represents a tenfold increase in sound intensity, but it's generally perceived by the human ear as a doubling of loudness. This logarithmic scaling allows for a more manageable and meaningful range of numbers to describe the vast array of sound intensities we can hear, from the faintest whisper to the roar of a jet engine.

#### Normalization in Practice

The use of decibels effectively normalizes the wide range of sound pressures into a scale that is more meaningful and practical for human interpretation and technical analysis. This normalization is crucial in various applications, from setting sound levels in audio engineering to assessing noise exposure in health and safety.

In audio processing and acoustic engineering, understanding and applying the concept of decibels is fundamental. It allows for the accurate and perceptually relevant measurement and manipulation of sound levels. This understanding is also key in designing audio equipment, architectural acoustics, and in the fields of audio forensics and environmental noise analysis.

In summary, the decibel system applies a logarithmic concept for normalization, making it an indispensable tool in audio and acoustic measurement. It bridges the gap between the physical properties of sound and the way sound is experienced by humans, ensuring that the measurements are as relevant and useful as possible.

### Understanding Sound Pressure Level (SPL)

Sound Pressure Level (SPL) is a critical concept in the study of acoustics and audio engineering, providing a quantitative measure of the pressure of a sound relative to a reference value. SPL is central to understanding how loud a sound is and is measured in decibels (dB), reflecting the logarithmic nature of human auditory perception. 

SPL is defined as the local pressure deviation from the ambient (average, or equilibrium) atmospheric pressure caused by a sound wave. In simpler terms, it measures the pressure variation a sound wave generates compared to the quiet or undisturbed air around it. SPL is typically measured using a sound level meter, which converts the physical pressure variations into electrical signals that can be quantified.

SPL is expressed in decibels, a logarithmic scale that compares the measured pressure with a reference pressure. The reference pressure in the case of SPL is usually the threshold of hearing, which is the quietest sound that the average human ear can detect, approximately 20 micropascals (µPa) in air. The decibel scale allows for a wide range of sound pressures to be represented in a compact and manageable scale. For example, a whisper might be around 30 dB, normal conversation around 60 dB, and a rock concert might be over 120 dB.

#### Why SPL Matters

Understanding and measuring SPL is crucial for several reasons:

1. **Health and Safety**: Prolonged exposure to high SPLs can lead to hearing damage or loss. Regulations often stipulate maximum SPLs to which individuals can be exposed in the workplace or in public spaces.

2. **Audio Quality**: In music production and audio engineering, SPL plays a significant role in achieving the desired sound quality and balance. 

3. **Environmental Considerations**: SPL measurements are used in environmental noise assessments, such as in assessing the impact of traffic noise on residential areas.

4. **Equipment Design**: Manufacturers of audio equipment, like speakers and microphones, use SPL measurements to design products that can handle or produce certain sound pressure levels without distortion.

In practical terms, SPL helps in setting appropriate levels in sound recording and live sound reinforcement. It aids in tuning and calibrating audio systems to desired levels and in designing spaces with specific acoustic properties. SPL measurements are also used in noise control engineering to develop noise reduction strategies and in the design of soundproofing materials and structures.

In summary, SPL is an indispensable tool in both the assessment and creation of sound. Whether it’s for protecting human hearing, creating audio art, or managing the acoustic environment, SPL measurements provide the objective data needed for informed decision-making and design in the world of sound and acoustics.

### Understanding the Short-Time Fourier Transform (STFT) in Audio Analysis

Now the star of the show.

![spectrogram.png](images%2Fspectrogram.png)

_The spectrogram of the `hello.wav` audio file in Adobe Audition._

The Short-Time Fourier Transform (STFT) is a fundamental technique in signal processing, particularly in the analysis of audio data. It serves as a bridge between the time domain and the frequency domain, providing a way to examine how the frequency content of a signal evolves over time. Let's break down this concept to understand its critical role in audio analysis and its application in computing spectrograms.

#### From Time Domain to Frequency Domain

Audio signals are typically represented in the time domain, where the signal's amplitude is plotted against time. While this representation is useful for understanding the overall amplitude variations over time, it doesn't offer insights into the frequency components that make up the signal. The Fourier Transform is a tool that converts a signal from the time domain to the frequency domain, revealing the different frequencies present in the signal. However, the standard Fourier Transform assumes the signal to be stationary, which is often not the case with audio signals that change over time.

#### The Essence of STFT

The Short-Time Fourier Transform addresses this limitation by dividing the longer time signal into shorter segments of equal length and then performing a Fourier Transform on each of these segments. This process provides a series of frequency spectra over time, offering a view into how the frequency content of the signal changes.

#### Computing the Spectrogram

The output of the STFT is often represented in a spectrogram, a visual representation of the spectrum of frequencies in a signal as they vary with time. In a spectrogram, the x-axis represents time, the y-axis represents frequency, and the intensity of each point represents the amplitude of a particular frequency at a given time. This visualization is particularly useful in various applications, from speech recognition to music analysis, as it provides a detailed view of the frequency dynamics within the audio signal.

#### Why STFT is Crucial

The STFT's ability to provide a time-varying frequency analysis makes it invaluable in audio processing. It allows for the examination of local frequency content and how it changes, which is essential in understanding and processing non-stationary signals like speech or music. By using STFT, we can extract and analyze the rich, temporal frequency characteristics of audio signals, which are crucial for various tasks in signal processing, audio engineering, and acoustic analysis.

In summary, the Short-Time Fourier Transform is a key technique in transforming time-domain audio data into a more informative frequency domain representation. Its application in computing spectrograms offers a powerful way to visualize and analyze the evolving frequency content of audio signals, providing deeper insights into the nature and characteristics of sound.

![izotope.png](images%2Fizotope.png)

_The spectrogram of the `hello.wav` audio file in iZotope RX._

The spectrogram of the `hello.wav` audio file, as displayed above, offers a comprehensive view of the audio data, encompassing both waveform and spectrogram representations. While traditional waveform views provide a sense of the overall dynamics and amplitude of the sound over time, they don't offer much detail about its frequency content. 

Advanced audio repair tools, such as those found in Adobe Audition or iZotope RX, capitalize on the spectrogram representation to effectively identify and eliminate noise elements in the audio. Unlike the waveform view, the spectrogram breaks down the sound into its constituent frequencies over time, making it easier to pinpoint specific frequencies that constitute noise or unwanted artifacts. 

These sophisticated tools allow audio professionals to visually isolate and remove these unwanted frequencies, which might be challenging to detect and address using only the waveform representation. By manipulating the audio data within the spectrogram, these tools can achieve a level of noise reduction and audio clarity that is difficult to accomplish with traditional waveform-based editing alone. This process underscores the importance of the spectrogram in modern audio processing, particularly in tasks requiring detailed and precise manipulation of the audio spectrum.

If you're just beginning to explore audio processing, I highly recommend experimenting with tools like Adobe Audition or iZotope RX. Try removing noises from your audio files using these applications, and you're likely to be astonished by the results. It almost feels like magic, watching unwanted sounds and disturbances vanish, but at its core, this is the power of mathematics and advanced algorithmic processing at work. These tools provide a hands-on experience of how sophisticated mathematical operations can transform and refine audio content.

In Photoshop, if you've ever used the patch or healing tools to repair a photo, you're engaging in a process that's conceptually similar to noise removal in audio files. Just as these tools in Photoshop allow you to seamlessly correct imperfections in an image, audio processing tools use comparable principles to identify and eliminate unwanted noise from sound recordings. Both processes involve a sophisticated blend of analysis and algorithmic manipulation to restore or enhance the original quality of the media.

## The Architecture of Whisper

Alright, having thoroughly covered all the prerequisite concepts, you are now fully equipped with the foundational knowledge necessary to delve deeply into the complexities of the Whisper model.

As we've seen, the Whisper model is a sophisticated blend of audio processing techniques and advanced neural network architectures, specifically leveraging the power of transformers and language models. Let's explore the architecture of the Whisper model in more detail.

We will be using the Apple MLX Example implementation of the Whisper model:

https://github.com/ml-explore/mlx-examples/tree/main/whisper

Here's the folder structure of the MLX implementation.

![folder-structure.png](images%2Ffolder-structure.png)

The folder showcases the structure of a Python implementation of the Whisper model, designed to be used within the MLX framework. Let's walk through the architecture and the role of each component file:

### `__init__.py`

```python
# Copyright © 2023 Apple Inc.

from . import audio, decoding, load_models
from .transcribe import transcribe

```

This file is typically empty and serves to indicate that the directory should be treated as a package in Python. It can also be used to initialize code for the package.

### `audio.py`

```python
# Copyright © 2023 Apple Inc.

import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import mlx.core as mx
import numpy as np

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        array = array[tuple(sl)]

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        pad_fn = mx.pad if isinstance(array, mx.array) else np.pad
        array = pad_fn(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    return mx.load(filename)[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x, window, nperseg=256, noverlap=None, nfft=None, axis=-1, pad_mode="reflect"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray],
    n_mels: int = 80,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array], shape = (*)
        The path to audio or either a NumPy or mlx array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    mx.array, shape = (80, n_frames)
        An  array that contains the Mel spectrogram
    """
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    if not isinstance(audio, mx.array):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))
    window = hanning(N_FFT)
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
    magnitudes = freqs[:-1, :].abs().square()

    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    mx.set_default_device(device)
    return log_spec

```

The `audio.py` module in the context of the Whisper model is dedicated to processing audio data. It encapsulates functions essential for preparing audio inputs to be fed into the neural network for further analysis or transcription. Here's an overview of its contents and their relevance:

- **Hyperparameters**: The module begins by defining several hyperparameters related to audio processing such as `SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, and others. These constants are used to configure how the audio is processed, including how it's sampled and how the spectrogram is computed.

- **`load_audio` function**: This function loads an audio file, resamples it to a specified sample rate if necessary, and returns a normalized waveform as a NumPy array. It utilizes `ffmpeg`, a powerful multimedia framework, to handle various audio formats and convert them into a uniform sample rate and mono channel.

- **`pad_or_trim` function**: It adjusts the length of an audio waveform to match a specific number of samples (`N_SAMPLES`). This is important for ensuring that the input to the neural network is consistent in size.

- **`mel_filters` function**: This function is responsible for loading the mel filterbank matrix. The Mel scale is used in audio processing to mimic the human ear's response to different frequencies. The filterbank is used to convert the frequency domain representation of the audio signal into the Mel scale, which is more meaningful for speech and music processing.

- **`hanning` function**: It generates a Hanning window array. Window functions are used in signal processing to minimize the edge effects in a frame of audio data when performing a Fourier transform.

- **`stft` function**: Short for Short-Time Fourier Transform, this function transforms time-domain audio data into the frequency domain. It's a critical step for audio analysis and is used to compute the spectrogram of the audio signal.

- **`log_mel_spectrogram` function**: This function computes the log-Mel spectrogram of an audio signal. The Mel spectrogram is a standard feature used in many audio processing tasks, including speech recognition, because it represents the sound in a way that's more closely aligned with human auditory perception.

Overall, this `audio.py` module is a collection of functions that pre-process audio data, converting it from its raw waveform into a format that's suitable for machine learning models to analyze—specifically, the log-Mel spectrogram, which is a common input representation for audio classification tasks in neural networks. The use of MLX functions within the module indicates that these operations are optimized for Apple Silicon, ensuring efficient computation.

### `decoding.py`

The `decoding.py` script in the MLX implementation of the Whisper model plays a crucial role in converting processed audio data into meaningful information. It consists of several key components and functionalities essential for the Whisper model's operation.

```python
# Copyright © 2023 Apple Inc.

import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def detect_language(
    model: "Whisper", mel: mx.array, tokenizer: Tokenizer = None
) -> Tuple[mx.array, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : mx.array, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != [model.dims.n_audio_ctx, model.dims.n_audio_state]:
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mx.array(mask)
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: mx.array
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Perform a forward pass on the decoder and return per-token logits"""
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        logits, self.kv_cache, _ = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits.astype(mx.float32)

    def rearrange_kv_cache(self, source_indices):
        """Update the key-value cache according to the updated beams"""
        # update the key/value cache to contain the selected sequences
        if source_indices != list(range(len(source_indices))):
            self.kv_cache = tree_map(lambda x: x[source_indices], self.kv_cache)

    def reset(self):
        self.kv_cache = None


class SequenceRanker:
    def rank(
        self, tokens: List[List[mx.array]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[List[int]]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : mx.array, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : mx.array, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        sum_logprobs: mx.array, shape = (n_batch)
            updated cumulative log probabilities for each sequence

        """
        raise NotImplementedError

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> Tuple[Sequence[Sequence[mx.array]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : mx.array, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : mx.array, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[mx.array]], length = n_audio
            sequence of mx.arrays containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            next_tokens = mx.random.categorical(logits=logits / self.temperature)

        next_tokens = mx.argmax(logits, axis=-1)
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1)

        current_logprobs = logprobs[mx.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        eot_mask = tokens[:, -1] == self.eot
        next_tokens = next_tokens * (1 - eot_mask) + self.eot * eot_mask
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = mx.all(tokens[:, -1] == self.eot)
        return tokens, completed, sum_logprobs

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        # make sure each sequence has at least one EOT token at the end
        tokens = mx.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class LogitFilter:
    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply any filtering or masking to logits

        Parameters
        ----------
        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int, n_vocab: int):
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        return logits + self.mask


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        mask = np.zeros(logits.shape, np.float32)
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            mask[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = sampled_tokens.tolist()
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    mask[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    mask[k, : self.tokenizer.eot] = -np.inf

            timestamps = [
                i for i, v in enumerate(seq) if v > self.tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                last_timestamp = timestamps[-1]
                if not last_timestamp or penultimate_was_timestamp:
                    last_timestamp += 1
                mask[k, self.tokenizer.timestamp_begin : last_timestamp] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            mask[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                mask[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = logits - mx.logsumexp(logits, axis=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                axis=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                mask[k, : self.tokenizer.timestamp_begin] = -np.inf

        return logits + mx.array(mask, logits.dtype)


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            raise NotImplementedError("Beam search decoder is not yet implemented")
            # self.decoder = BeamSearchDecoder(
            #    options.beam_size, tokenizer.eot, self.inference, options.patience
            # )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin, model.dims.n_vocab)
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens(), model.dims.n_vocab)
            )
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array):
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: mx.array, tokens: np.array):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                # write language tokens
                tokens[:, self.sot_index + 1] = np.array(lang_tokens)

        return languages, lang_probs

    def _main_loop(self, audio_features: mx.array, tokens: mx.array):
        n_batch = tokens.shape[0]
        sum_logprobs: mx.array = mx.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = mx.softmax(
                        logits[:, self.sot_index].astype(mx.float32), axis=-1
                    )
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logits = logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed, sum_logprobs = self.decoder.update(
                    tokens, logits, sum_logprobs
                )

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.reset()

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: mx.array) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: mx.array = self._get_audio_features(mel)  # encoder forward pass
        tokens: np.array = np.array(self.initial_tokens)
        tokens = np.broadcast_to(tokens, (n_audio, len(self.initial_tokens))).copy()

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat tokens by the group size, for beam search or best-of-n sampling
        tokens = mx.array(tokens)
        if self.n_group > 1:
            tokens = tokens[:, None, :]
            tokens = mx.broadcast_to(
                tokens, [n_audio, self.n_group, len(self.initial_tokens)]
            )
            tokens = tokens.reshape(
                tokens, (n_audio * self.n_group, len(self.initial_tokens))
            )

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = tokens[..., self.sample_begin :].tolist()
        tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: mx.array, shape = (80, 3000) or (*, 80, 3000)
        An array containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result

```

### Core Functions and Classes

- **`compression_ratio` Function**: This function calculates the compression ratio of a given text by comparing its original size to its size after compression. This can be used to assess the efficiency of text compression.

- **`detect_language` Function**: This function determines the spoken language in the audio and returns the most probable language tokens along with their probability distribution. It's crucial for models that need to recognize or process audio in multiple languages.

- **`DecodingOptions` Data Class**: This class defines various options for the decoding process, such as the task type (transcribe or translate), language, sampling options, and others. It allows for customization of the decoding process to suit different requirements.

- **`DecodingResult` Data Class**: This class is used to store the results of the decoding process, including audio features, detected language, text, and other relevant metrics. 

- **`Inference` Class**: This class handles the forward pass through the decoder, managing operations like caching to improve efficiency.

- **`SequenceRanker` and `MaximumLikelihoodRanker` Classes**: These classes are used to rank sequences of sampled tokens based on their log probabilities, enabling the selection of the most likely sequence.

- **`TokenDecoder` and `GreedyDecoder` Classes**: These are responsible for determining the next tokens based on the current context and logits, playing a key role in the autoregressive generation of text.

- **`LogitFilter`, `SuppressBlank`, `SuppressTokens`, and `ApplyTimestampRules` Classes**: These classes apply various rules to suppress or penalize certain tokens during the decoding process, ensuring more accurate and contextually relevant outputs.

- **`DecodingTask` Class**: This is the main class that orchestrates the decoding process, integrating various components like token decoders, logit filters, and sequence rankers to decode the audio input effectively.

- **`decode` Function**: This high-level function interfaces with the `DecodingTask` class to perform the decoding of audio segments, provided as Mel spectrograms. It's the primary entry point for using the decoding functionalities in the model.

The `decoding.py` module encapsulates the complex process of translating audio data into text or other forms of output. It demonstrates the integration of various machine learning components, such as language detection, token decoding, and sequence ranking, to process audio input efficiently. This module is a testament to the intricate work behind speech recognition and language processing tasks in modern AI systems, showcasing the depth and sophistication of the Whisper model.

### `load_models.py`

The `load_models.py` script in the MLX implementation of the Whisper model is designed to load the model with its necessary configurations and weights. It plays a crucial role in initializing the Whisper model for use in various tasks like speech recognition or language translation.

```python
# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.update(weights)
    mx.eval(model.parameters())
    return model

```

- **Importing Required Modules**: The script begins by importing necessary Python packages and modules, including `mlx.core` and `mlx.nn` for MLX framework functionalities, `json` for handling configuration files, and `huggingface_hub` for downloading model weights from Hugging Face's model repository.

- **`load_model` Function**: This is the primary function of the script, responsible for loading the Whisper model.

  - **Path Handling**: The function first checks if the provided `path_or_hf_repo` (a local path or Hugging Face repository ID) exists locally. If not, it uses `snapshot_download` from `huggingface_hub` to download the model from the specified repository.

  - **Configuration Loading**: It then loads the model configuration from the `config.json` file. This file contains essential parameters for the model, such as its dimensions and quantization details.

  - **Loading Weights**: The model's weights are loaded from a `.npz` file using `mx.load`. These weights are then organized (or 'unflattened') to match the structure expected by the model.

  - **Model Initialization**: The `Whisper` model is instantiated with the loaded configuration and set to the specified data type (dtype).

  - **Applying Quantization (if applicable)**: If quantization parameters are present in the configuration, they are applied to the model using `nn.QuantizedLinear.quantize_module`. Quantization can help optimize the model for performance, especially on specific hardware.

  - **Updating Model with Weights**: The model's parameters are updated with the loaded weights.

  - **Enabling Lazy Evaluation with `mx.eval`**: In the final step of the `load_model` function, the model is set to evaluation mode using `mx.eval`. This is a critical step in MLX, as it activates the framework's lazy evaluation feature. Lazy evaluation means that computations are not executed immediately when they are defined but are deferred until their results are actually needed.

  - **Implications for Inference**: By setting the model to evaluation mode, all subsequent operations on the model are optimized for inference. In this mode, MLX efficiently manages the computation graph, ensuring that unnecessary calculations are avoided and that the model runs as efficiently as possible. This is particularly important for inference tasks, where speed and resource optimization are crucial.

  - **Consistent Model Behavior**: This step also ensures consistent behavior of the model, particularly important for tasks like speech recognition or language translation where consistent and predictable outputs are key. 

The `load_models.py` script is an integral part of the Whisper model's infrastructure, handling the critical task of preparing the model for use. By efficiently managing the loading of configurations, weights, and ensuring the model is in the correct state, it sets the stage for performing complex audio processing tasks. This script exemplifies the necessary preparations required in machine learning models for their effective deployment and utilization in real-world applications.

### `timing.py`

The `timing.py` script in the MLX implementation of the Whisper model plays a specialized role in determining the timing of words in the transcribed audio. This script is particularly important for applications that require accurate word-level timestamps, such as subtitle generation or detailed audio analysis.

```python
# Copyright © 2023 Apple Inc.

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx
import numba
import numpy as np
from scipy import signal

from .audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .model import Whisper


def median_filter(x: np.ndarray, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        # F.pad requires the padding width to be smaller than the input dimension
        return x

    if (ndim := x.ndim) <= 2:
        # `F.pad` does not support 1D or 2D inputs for reflect padding but supports 3D and 4D
        x = x[None, None, :]

    assert (
        filter_width > 0 and filter_width % 2 == 1
    ), "`filter_width` should be an odd number"

    x = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")

    # todo: more efficient version in mlx
    result = signal.medfilt(x.astype(np.float32), kernel_size=(1, 1, filter_width))[
        ..., pad_width:-pad_width
    ]

    if ndim <= 2:
        result = result[0, 0]

    return result


@numba.jit(nopython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


@numba.jit(nopython=True, parallel=True)
def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)


def dtw(x: np.ndarray) -> np.ndarray:
    # todo: more efficient version in mlx
    return dtw_cpu(x)


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: mx.array,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    if len(text_tokens) == 0:
        return []

    tokens = mx.array(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    )

    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens[None, :])
    # consider only the logits associated with predicting text
    sampled_logits = logits[0][len(tokenizer.sot_sequence) : -2, : tokenizer.eot]
    token_probs = mx.softmax(sampled_logits.astype(mx.float32), axis=-1).astype(
        sampled_logits.dtype
    )
    text_token_probs = mx.take_along_axis(
        token_probs, mx.array(text_tokens)[:, None], axis=1
    ).squeeze(1)
    text_token_probs = np.array(text_token_probs)

    # heads * tokens * frames
    weights = mx.stack(
        [cross_qk[_l.item()][0, _h.item()] for _l, _h in model.alignment_heads]
    )
    weights = weights[:, :, : num_frames // 2]
    weights = mx.softmax(weights * qk_scale, axis=-1)
    mean = mx.mean(weights, axis=-2, keepdims=True)
    std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
    weights = (weights - mean) / std
    weights = median_filter(np.array(weights), medfilt_width)

    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: mx.array,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=timing.probability,
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

```

- **`median_filter` Function**: This function applies a median filter to an input array, which is helpful in smoothing out the values along the last dimension of the array. Median filtering is often used in signal processing to reduce noise while preserving edges.

- **`backtrace` Function**: Utilizing the Numba JIT (Just-In-Time) compiler for performance optimization, this function traces back through a matrix (commonly used in dynamic programming algorithms like the Viterbi algorithm) to find the optimal path or sequence of states.

- **`dtw_cpu` and `dtw` Functions**: These functions implement the Dynamic Time Warping (DTW) algorithm. DTW is a method used to align sequences that may vary in time or speed. In the context of audio processing, it's particularly useful for aligning spoken words with their corresponding timestamps.

- **`WordTiming` Data Class**: This class serves as a structured way to represent the timing information of words, including the start and end times and the probability of the word.

- **`find_alignment` Function**: This function is central to the script. It finds the alignment between the transcribed text tokens and the audio features. It uses the model's forward pass with cross-attention queries and keys (`cross_qk`) to obtain weights for each token-frame pair, which are then processed to calculate the timing for each word.

- **`merge_punctuations` Function**: This function adjusts the timing of words by merging punctuations that are typically appended or prepended to words, ensuring that punctuation is accurately represented in the timing data.

- **`add_word_timestamps` Function**: This function takes the segments of transcribed text and adds word-level timestamps to them. It handles the integration of the timing data into the final transcription output, considering nuances like word duration and segment boundaries.

The `timing.py` script is a sophisticated piece of the Whisper model's architecture, focusing on the temporal aspect of speech transcription. It illustrates the complexity involved in accurately mapping the spoken words to their corresponding timestamps in the audio stream. This functionality is vital for applications where understanding the exact timing of each word is as important as the words themselves, such as in synchronized subtitles or detailed linguistic analysis. The script showcases the application of advanced algorithms and techniques in the field of speech processing, further highlighting the Whisper model's capabilities in handling complex audio transcription tasks.

### `tokenizer.py`

The `tokenizer.py` script in the MLX implementation of the Whisper model is responsible for handling all aspects of tokenization, which is the process of converting text into a sequence of tokens that can be processed by the model. This script is crucial for both encoding input text and decoding model outputs into human-readable language. 

```python
# Copyright © 2023 Apple Inc.

import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    with open(vocab_path) as fid:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in fid if line)
        }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )

```

- **Language Support**: The script defines a dictionary named `LANGUAGES` that maps language codes to their corresponding language names. This extensive list indicates the multilingual capabilities of the Whisper model.

- **Tokenizer Class**: The core of the script is the `Tokenizer` class. This class serves as a wrapper around the `tiktoken` library and provides quick access to special tokens and functions for encoding and decoding text.

- **Encoding and Decoding Methods**: The class includes methods to encode text into tokens (`encode`) and decode tokens back into text (`decode`). These methods are essential for processing the input and output of the Whisper model.

- **Special Tokens**: The `Tokenizer` class handles various special tokens, such as end-of-text (`eot`) and start-of-transcript (`sot`), which are crucial for indicating the beginning and end of sequences in the model.

- **Language and Task Configuration**: The tokenizer can be configured for specific languages and tasks (like transcription or translation), indicating its flexibility and adaptability for different use cases.

- **Word Alignment and Timestamps**: Additional functions in the script, such as `split_to_word_tokens`, are designed for aligning words with their corresponding timestamps, an important feature for applications like subtitle generation.

- **Cached Encoding**: The `get_encoding` function, with its use of `lru_cache`, efficiently retrieves the encoding configuration, ensuring that the tokenizer is fast and responsive.

- **Language Code Lookup**: The script also includes functionality for converting language names to their respective codes, supporting a user-friendly interface for language selection.

The `tokenizer.py` script is a vital component of the Whisper model's architecture, enabling the model to interact with and process language data effectively. By handling the complexities of tokenization and language encoding, this script facilitates the model's ability to understand and generate human language, making it a key player in the model's overall functionality.

### `transcribe.py`

The `transcribe.py` script in the MLX implementation of the Whisper model provides a high-level interface for transcribing audio into text. This script is designed to integrate various components of the Whisper model to deliver a complete transcription solution. 

```python
# Copyright © 2023 Apple Inc.

import sys
import warnings
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .load_models import load_model
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer


def _format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


class ModelHolder:
    model = None
    model_path = None

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        if cls.model is None or model_path != cls.model_path:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
        return cls.model


def transcribe(
    audio: Union[str, np.ndarray, mx.array],
    *,
    path_or_hf_repo: str = "mlx-community/whisper-tiny",
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array]
        The path to the audio file to open, or the audio waveform

    path_or_hf_repo: str
        The localpath to the Whisper model or HF Hub repo with the MLX converted weights.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
    model = ModelHolder.get_model(path_or_hf_repo, dtype)

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":
            make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
                system_encoding
            )
        else:
            make_safe = lambda x: x

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` decoding option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def decode_with_fallback(segment: mx.array) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return decode_result

    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    input_stride = N_FRAMES // model.dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *, start: float, end: float, tokens: mx.array, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    with tqdm.tqdm(
        total=content_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)
            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
            mel_segment = mel[seek : seek + segment_size]
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(mel_segment)
            tokens = np.array(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            previous_seek = seek
            current_segments = []

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: Optional[dict]) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: List[dict]) -> Optional[dict]:
                return next((s for s in segments if s["words"]), None)

            timestamp_tokens = tokens >= tokenizer.timestamp_begin
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = np.where(
                np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
            )[0]
            consecutive += 1
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero()[0]]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek += segment_size

            if word_timestamps:
                add_word_timestamps(
                    segments=current_segments,
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_frames=segment_size,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )

                if not single_timestamp_ending:
                    last_word_end = _get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * FRAMES_PER_SECOND)

                # skip silence before possible hallucinations
                if hallucination_silence_threshold is not None:
                    threshold = hallucination_silence_threshold
                    if not single_timestamp_ending:
                        last_word_end = _get_end(current_segments)
                        if last_word_end is not None and last_word_end > time_offset:
                            remaining_duration = window_end_time - last_word_end
                            if remaining_duration > threshold:
                                seek = round(last_word_end * FRAMES_PER_SECOND)
                            else:
                                seek = previous_seek + segment_size

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(
                                current_segments[si + 1 :]
                            )
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"])
                                    * FRAMES_PER_SECOND
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = _get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end

            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
                    print(make_safe(line))

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )

```

- **Formatting Timestamps**: The script includes a utility function `_format_timestamp` to convert time in seconds to a formatted timestamp string. This is useful for producing human-readable timestamps in the transcribed text.

- **Model Holder Class**: The `ModelHolder` class acts as a singleton to load and hold an instance of the Whisper model. This approach ensures efficient use of resources, as the model is loaded only once and reused for multiple transcriptions.

- **Transcribe Function**: The core of the script is the `transcribe` function. It orchestrates the process of transcribing audio, handling different input types (file paths or audio arrays), and setting various transcription options.

    - **Mel Spectrogram Conversion**: The function begins by converting the input audio into a mel spectrogram representation, which is a critical step for audio processing in the Whisper model.
  
    - **Language Detection**: If the language of the audio is not specified, the script includes functionality to automatically detect the language, facilitating multilingual transcription capabilities.
  
    - **Transcription with Fallback**: The transcription process is designed to handle varying conditions (like temperature settings) and apply fallback strategies based on factors like compression ratio and log probability thresholds. This ensures robustness in the transcription output.

    - **Word Timestamps**: If enabled, the script can generate word-level timestamps, which are valuable for applications like subtitle generation.

    - **Handling Punctuations**: The script includes functions to correctly process and align punctuation with the transcribed words.

- **Decoding Options**: The script allows customization of the transcription process through various decoding options, providing flexibility to cater to different use cases and requirements.

- **Return Structure**: The output of the `transcribe` function is a dictionary containing the transcribed text, detailed segment-level information, and the detected language, offering a comprehensive view of the transcription result.

The `transcribe.py` script is a crucial component that brings together the different elements of the Whisper model to provide a practical and flexible transcription tool. Its ability to handle various input types, languages, and transcription nuances makes it a versatile solution for converting audio into text. This script demonstrates the integration of complex model components and audio processing techniques to deliver a user-friendly and efficient transcription service.

### `whisper.py`

The `whisper.py` script in the MLX implementation of the Whisper model is a comprehensive module that encapsulates the model architecture and key functionalities. This script is designed to integrate various components, providing a cohesive structure for the Whisper model's operations.

```python
# Copyright © 2023 Apple Inc.

import base64
import gzip
import math
from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k, v = kv_cache

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.astype(mx.float32)

        w = mx.softmax(qk, axis=-1).astype(q.dtype)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x += y
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x += y
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))).astype(x.dtype))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = LayerNorm(n_state)

    def __call__(self, x):
        x = nn.gelu(self.conv1(x)).astype(x.dtype)
        x = nn.gelu(self.conv2(x)).astype(x.dtype)
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(self, x, xa, kv_cache=None):
        """
        x : mx.array, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : mx.array, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )

        x = self.ln(x)
        return x @ self.token_embedding.weight.T, kv_cache, cross_qk


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        if isinstance(dump, np.ndarray):
            self.alignment_heads = mx.array(dump)
        elif isinstance(dump, bytes):
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self.alignment_heads = mx.array(np.asarray(mask.nonzero()).T)
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel))
        return logits, cross_qk

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    decode = decode_function

```


### Overview of `whisper.py`

- **ModelDimensions Data Class**: Defines the structural dimensions of the Whisper model, including parameters for audio and text processing like the number of mel spectrogram features (`n_mels`), context size, state size, number of heads, and layers.

- **Sinusoids Function**: Generates sinusoidal embeddings for positional encoding, crucial for maintaining the sequence order of inputs in the model.

- **LayerNorm Class**: A custom layer normalization class tailored for the Whisper model, ensuring that the neural network layers function optimally by normalizing the inputs.

- **MultiHeadAttention Class**: Implements the multi-head attention mechanism, allowing the model to focus on different parts of the input sequence, which is vital for understanding and generating language.

- **ResidualAttentionBlock Class**: Combines multi-head attention with residual connections, an essential part of the model's architecture that helps in learning deeper representations without losing information from earlier layers.

- **AudioEncoder and TextDecoder Classes**: These are the core components for processing audio and text data. They incorporate multiple layers of attention and other neural network layers to process the input data effectively.

- **Whisper Class**: This is the main class of the script, bringing together all the components to form the complete Whisper model. It defines the model's forward pass, combining the audio encoder and text decoder to process input data and generate outputs.

- **Utility Functions and Methods**: The script includes functions such as `set_alignment_heads` for configuring specific attention heads and methods like `embed_audio`, `logits`, and `forward_with_cross_qk` for processing audio and generating logits.

- **Integration with Decoding Functions**: The script integrates with other components of the Whisper model, such as `detect_language` and `decode` functions, to provide language detection and decoding capabilities.

The `whisper.py` script is a central piece in the MLX implementation of the Whisper model, encompassing the model's architecture and essential functionalities. It demonstrates a comprehensive integration of various neural network components and techniques, tailored for efficient and effective audio processing and language understanding. This script exemplifies the advanced capabilities of the Whisper model in handling speech recognition and language processing tasks.

### `torch_whisper.py`

The `torch_whisper.py` script serves solely for testing and validation. It's not a part of the operational MLX implementation of the Whisper model. The purpose of this script is to facilitate the testing of the Whisper model's PyTorch version, as used in the `test.py` script. Specifically, it allows for a comparative analysis and performance benchmarking between the PyTorch version and the MLX implementation of the model. This approach ensures the consistency and efficiency of the MLX version by validating it against the established PyTorch implementation.

## Menny, the Sonic Whisperer

For those interested in exploring the capabilities of AI-driven audio transcription, I present two demo versions. The first, a more comprehensive and feature-rich chatbot named _PippaGPT-MLX_, can be found here:

[PippaGPT-MLX](https://github.com/neobundy/pippaGPT-MLX)

This version, while intricate and offering a GPT-4 level experience with features like voice chat, is not my current focus. Therefore, I encourage you to view it as an exploratory project on your own. It's designed to be fully operational right from the start. To unlock the full voice chat experience, an ElevenLabs API key is required.

On a simpler note, I introduce _Menny, the Sonic Whisperer_. This demo leverages the MLX implementation of the Whisper model to transcribe spoken words into text. It serves as an accessible entry point for those looking to delve into audio processing with MLX.

To get started with Menny, you'll need to initiate the `audio_server.py` script to activate the audio server:

```bash
python audio_server.py
```

Following that, launch the `menny-the-sonic-whisperer.py` script with Streamlit:

```bash
streamlit run menny-the-sonic-whisperer.py
```

With these steps, _Menny, the Sonic Whisperer_, is all set to transcribe your voice into text, showcasing the practical application of the Whisper model within the MLX framework.

And there we have it—a comprehensive exploration of digital audio processing and the Whisper model.

I cannot emphasize enough the importance of embracing the journey of learning. My approach has always been to understand and apply concepts through an object-oriented lens, and this repository is yet another iteration of that practice.

Remember, the true value lies in the journey, not just the destination.
