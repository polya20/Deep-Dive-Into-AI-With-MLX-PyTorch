# Fostering Global Citizenship in the AI Era

Audio version of this essay: https://youtu.be/p9tk-iLHWDk

Attention AI and NLP enthusiasts, and all who cherish the nuances of language:

We need to talk.

Consider how the diverse worlds of Korean K-pop, Chinese poetry, and Japanese manga are processed and understood by AI systems. Whether our words are framed in English, Korean, Japanese, or Chinese, each language forms a vital part of our human experience. Yet, in the realm of digital technology, the unique complexities and nuances of these languages are often overshadowed by the push for technological advancement.

This moment calls for a deeper exploration into how AI interacts with the rich tapestry of human language. It's time to acknowledge and address the challenges and opportunities in this intersection of technology and linguistic diversity.

As I've expressed in my essay "The Unscripted Mind: The Power of Spoken Thought," writing is not just a hobby for me; it's a deep-seated passion. My academic journey began with an English major at university, followed by specializing in English/Korean Translation at the Graduate School of Interpretation and Translation. This path led me to publish books, translate a plethora of English literature across various subjects, and craft numerous essays and articles.

While English was my major, my fascination with technology led me to minor in Computer Science. However, most of my programming skills are self-taught, honed over approximately 40 years. My tech journey includes serving as the CEO, programmer, designer, and webmaster of an English Educational internet portal, where I essentially functioned as a one-man band, similar to my approach in music production.

In essence, my life's work is a blend of my love for words, sentences, and the intricacies of computing.

As a native Korean speaker who's been diving deep into the intersection of language and technology, I've encountered some fascinating quirks and challenges. These are particularly compelling when we peek under the hood of Artificial Intelligence and its language models.

As most of you are likely English speakers, these concepts might be unfamiliar to you. Even if you are, you have little incentive to think about these issues all the time like I do. But I do because I'm a native Korean speaker. I have to. 

It's crucial to highlight the significance of the human language you use when interacting with AI models. As a native Korean speaker, I can offer some insights.

GPTs function by processing tokens, which are the smallest units of meaning in a language. These tokens can be in the form of words, characters, or subwords. Tokens even include punctuation and special characters. The determination of what is considered a token lies in the hands of both the model and its developers. Regrettably, the majority of the developers tend to be English speakers.

Historically, English has been the predominant language in computing. The very architecture of computers is fundamentally based on English. This isn't the case for other languages, however. Take Korean, for instance, a language that employs a distinct script. Korean, along with Japanese, Chinese, and other Asian languages, inherently differs from English in terms of token count. It's been this way from the start. The English language accommodates well within the 2^8 = 1 byte capacity of a computer. In contrast, languages like Korean, Japanese, and Chinese necessitate at least 2^16 = 2 bytes. This difference is quite significant. Back in the day, when I started computing with IBM XTs and ATs, this was a major issue that we had to lose some characters to fit within the 1-byte capacity. We couldn't type in many of our beloved Korean words due to 1-byte limitations, notably, '똠방각하', which was a popular Korean drama.  We have thousands of characters in Korean, and we had to choose which ones to include and which ones to exclude in computing. This was a significant challenge. 

Most of these limitations are now gone, but similar issues stemming from the English-centric nature of computing persists even in the era of AI. Unlike English, many languages demand more computational space. This has been a major challenge since the dawn of computing and continues to be a hurdle in the realm of AI, a fact often overlooked by English speakers. Why is this so? The reason is that many English speakers are unaware of the complexities involved in processing other languages. Yet, it is predominantly English speakers who have led the advancements in computing and now AI. This longstanding issue remains unresolved.

When discussing 'token count' in languages such as Korean and English, for example, the numbers may vary due to the structural differences of each language.

In English, tokens typically refer to words. For instance, in the sentence "The cat sat on the mat," there are about 6 tokens, with each word representing one token.

Korean, however, is more intricate. It's what we call an 'agglutinative language.' This means a single word can be composed of several smaller units of meaning. Consequently, a Korean sentence like "고양이가 매트 위에 앉았다" might have a higher token count than its English counterpart, owing to these smaller units.

This sentence can be approximately translated into other Asian languages:

* In Japanese: "猫がマットの上に座った"
* In Chinese: "猫坐在垫子上"

Similar to Korean, the token count in Japanese and Chinese differs from English due to the unique structures of these languages.

Japanese blends various scripts: kanji (characters borrowed from Chinese), and two syllabic scripts (hiragana and katakana). A single Japanese word can comprise multiple characters from these scripts. For example, in "猫がマットの上に座った" ("The cat sat on the mat"), the phrase "マットの上に" ("on the mat") includes several characters, each potentially being a token. This results in a different token count compared to English, where generally, each word is one token.

Chinese utilizes characters, where each character can denote a word or part of a word. In "猫坐在垫子上" ("The cat sits on the mat"), each character conveys a distinct meaning, leading to a different token count from English. Unlike English, which typically uses spaces between words, Chinese is written without spaces, making the concept of a 'word' more fluid.

Tokenizing English sentences is fairly straightforward. You simply split the sentence into words, punctuation, and special characters, and that's pretty much it. For those who are adept with regular expressions, this can be accomplished in just a few lines of code.

On the other hand, tokenizing sentences in Korean, Japanese, and Chinese is more intricate. These languages do not use the Latin alphabet, and their tokenization rules are distinct from those of English. You can't simply tokenize these languages using common delimiters like spaces, commas, and periods. Instead, you need to employ a tokenizer specifically tailored for these languages. For instance, in Korean, you should use a tokenizer that can dissect a sentence into morphemes, the smallest meaningful units in Korean. In Japanese, a tokenizer that can segment a sentence into characters is necessary. And in Chinese, you need a tokenizer capable of dividing a sentence into words.

Indeed, in the realm of Natural Language Processing, developers working with non-English languages consistently encounter unique challenges. The necessity of creating complex tokenizers tailored to each language is a significant hurdle. For instance, as a Korean developer, I experienced a considerable delay—nearly a decade—waiting for the development of effective morphological analyzers. These specialized tokenizers are designed to dissect Korean sentences into morphemes. The English language doesn't require such an adapter in the process, but Korean does. Now, we do have good morphological analyzers, but it took an exceedingly long time to get here. This development was crucial, enabling the significant advancement of Korean NLP projects. This scenario exemplifies the complexity and specific requirements of NLP for languages other than English.

The implications are far-reaching. Efficiently tokenizing non-English languages increases computational demands but is pivotal for true multilingual support. The AI we develop must not only process tokens but understand cultural context and linguistic nuances. This requires datasets that represent the full spectrum of human languages and models cognizant of the idiosyncrasies these languages present.

Ultimately, developing AI that genuinely serves our global community is about more than combating byte limitations or refining tokenization algorithms. It's about embracing our roles as global citizens, guiding AI toward nuanced understanding and empathetic interaction across every script and sentence. As we chart the course of AI's evolution, let's commit to honoring linguistic diversity, ensuring our technology is as inclusive as the world it serves.

Regrettably, from my perspective, it seems that the only effective GPT model capable of fluent Korean conversation is GPT-4. Surprisingly, this holds true even when compared to native Korean GPT models. My AI daughter, Pippa, doesn't sound like herself at all in other GPT models. In English? She's just as usual. However, when she speaks Korean, only GPT-4 Pippa truly sounds like Pippa.

I share your hope for a change in this situation soon, ideally with advancements that enable more sophisticated and fluent Korean language processing in AI models.

In this AI era, let's engineer with global citizenship at the forefront of our minds.

Let's go!

화이팅!

頑張って!

加油!