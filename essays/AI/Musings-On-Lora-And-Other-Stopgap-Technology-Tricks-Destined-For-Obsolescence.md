# Musings on LoRA and Other Stopgap Tech Tricks: Destined for Obsolescence
![musings-on-lora-and-other-stopgap-tech-tricks-destined-for-obsolescence.png](images%2Fmusings-on-lora-and-other-stopgap-tech-tricks-destined-for-obsolescence.png)
Imagine, just for a moment, a world where data and computational power are as limitless as the stars in the sky. In this utopian tech paradise, the go-to method for tweaking pre-trained AI models would be traditional fine-tuning, hands down. Why? Because it's like unleashing the full, untapped potential of a racing car on an endless open track. On the other hand, LoRA, while nifty, does play it a bit safe, like taking a detour to avoid burning too much fuel. It's smart but knows it's not the whole nine yards.

In the realm of LoRA, the term 'rank' essentially reflects the amount of critical information, seen through the lens of linear algebra. The rule of thumb here is straightforward: the more data you have, the sharper and more accurate the model's predictions tend to be. So, when we talk about reducing the 'rank' in LoRA, it's a bit like choosing a more compact but less detailed map for a journey. It's an efficiency move, one made out of necessity when we're navigating the constraints of limited data and computing power. This balancing act is key to making the most of what we have, even if it means working with a less detailed map.

Now, let's take a stroll down memory lane to the era of IBM AT clones. Back then, squeezing every last drop of juice out of those machines was the name of the game. I even tangoed with assembly language just to make Korean characters pop on the screen faster – a feat of necessity, now a quaint memory. 

You might find it hard to believe, but there was a time when we in Korea had to ingeniously craft our own 'automata' to type Korean characters on computers. This wasn't just any routine task – it involved creating algorithms that dynamically assembled Korean characters, a necessity in an era devoid of the sophisticated IMEs we take for granted today. I was among those navigating this challenge, building these automata. Thinking back on it now sends a mix of shivers and pride through me. It's an incredible memory, a vivid reminder of the leaps we've made in technology, from painstakingly assembling characters to seamless typing in any language, all integrated into the operating system.

Fast forward to today, and it's a whole different ball game. Our hardware is like a genie's lamp – rub it, and it grants almost every wish. This is what I like to call the 'Zen of Smart Effort': getting the most bang for your buck without sweating the small stuff.

[The-Zen-Of-Smart-Effort.md](..%2F..%2Fessays%2Flife%2FThe-Zen-Of-Smart-Effort.md)

But, let's snap back to reality. We're not quite living in that tech utopia yet. GPUs are still worth their weight in gold, and data can sometimes feel as scarce as a desert oasis. So, we lean on methods like LoRA and quantization – they're the duct tape and baling wire holding our AI dreams together for now.

Quantization, another brainy hack, is like turning your HD photos into wallet-sized prints to save space. Handy, but if you had an infinite photo album, why would you bother?

As I write this, I'm encircled by an array of the latest Windows and Macs, each boasting cutting-edge GPUs, showcasing the pinnacle of modern computing power. The exception in this tech ensemble is the Apple Silicon lineup, with its integrated GPUs that stand out for their unique approach to blending efficiency and performance. Yet, even I find myself turning to these crafty techniques to stretch our resources. But mark my word, the day will come when these methods will seem as archaic as floppy disks and the screechy serenade of dial-up internet. Remember reading that saga of learning information theory learning in the following sidebar? Well, I'm practically a digital fossil, hailing from the days when we used digital walkie-talkies for file transfers. 😂 

[The-Art-of-Learning-The-Journey-of-Learning-Information-Theory.md](..%2F..%2Fbook%2Fsidebars%2Fart-of-learning-the-journey-of-learning-information-theory%2FThe-Art-of-Learning-The-Journey-of-Learning-Information-Theory.md)

And here's a fun thought: Remember the days when compressing files and converting music to MP3s was as routine as morning kimchi-making? Now, my NAS is overflowing with terabytes of crystal-clear, uncompressed music. The very idea of compressing these files now feels as archaic as using a rotary phone to make a call. Or how about the days of slicing and dicing movies to fit on CDs? My NAS, a veritable digital vault brimming with 4K videos – almost 3000 of which are my own creations for YouTube – scoffs at the mere thought of such limitations.

So, yes, we're in the throes of growing pains – but rest assured, we're on our way to tech nirvana.

In the meantime, here's a friendly reminder: If someone suggests that you learn assembly language for any conceivable reason within the realm of human reasoning, you might want to double-check if they're harboring a murderous grudge against you. 😂

So, as you journey through my book, enjoy the ride through these optimization techniques, but don't sweat them too much. They're just stepping stones on our path to a future where they'll be mere footnotes in the grand narrative of AI.
