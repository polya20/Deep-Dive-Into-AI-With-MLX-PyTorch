explain further on how sampler works. please make sure to explain what we mean by taking samples from distributions in the context of diffusion models:

from `stable_diffusion\sampler.py`:

class SimpleEulerSampler:
    """A simple Euler integrator that can be used to sample from our diffusion models.

    The method ``step()`` performs one Euler step from x_t to x_t_prev.
    """

    def __init__(self, config: DiffusionConfig):
        # Compute the noise schedule
        if config.beta_schedule == "linear":
            betas = _linspace(
                config.beta_start, config.beta_end, config.num_train_steps
            )
        elif config.beta_schedule == "scaled_linear":
            betas = _linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.num_train_steps
            ).square()
        else:
            raise NotImplementedError(f"{config.beta_schedule} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = mx.cumprod(alphas)

        self._sigmas = mx.concatenate(
            [mx.zeros(1), ((1 - alphas_cumprod) / alphas_cumprod).sqrt()]
        )

    @property
    def max_time(self):
        return len(self._sigmas) - 1

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        noise = mx.random.normal(shape, key=key)
        return (
            noise * self._sigmas[-1] * (self._sigmas[-1].square() + 1).rsqrt()
        ).astype(dtype)

    def add_noise(self, x, t, key=None):
        noise = mx.random.normal(x.shape, key=key)
        s = self.sigmas(t)
        return (x + noise * s) * (s.square() + 1).rsqrt()

    def sigmas(self, t):
        return _interp(self._sigmas, t)

    def timesteps(self, num_steps: int, start_time=None, dtype=mx.float32):
        start_time = start_time or (len(self._sigmas) - 1)
        assert 0 < start_time <= (len(self._sigmas) - 1)
        steps = _linspace(start_time, 0, num_steps + 1).astype(dtype)
        return list(zip(steps, steps[1:]))

    def step(self, eps_pred, x_t, t, t_prev):
        sigma = self.sigmas(t).astype(eps_pred.dtype)
        sigma_prev = self.sigmas(t_prev).astype(eps_pred.dtype)

        dt = sigma_prev - sigma
        x_t_prev = (sigma.square() + 1).sqrt() * x_t + eps_pred * dt

        x_t_prev = x_t_prev * (sigma_prev.square() + 1).rsqrt()

        return x_t_prev


from `stable_diffusion\__init__.py`:

class StableDiffusion:
    """
    Class for Stable Diffusion model.
    """
    def __init__(self, model: str = _DEFAULT_MODEL, float16: bool = False):
        """
        Initialize the StableDiffusion model with the given model and data type.

        float16: Whether to use float16 data type for half-precision.
        This can reduce the memory requirements of the model by half compared to 32-bit floating point numbers, at the cost of reduced numerical precision.

        """

        debug_print(f"Using Huggingface model: {model}")


        self.dtype = mx.float16 if float16 else mx.float32
        self.diffusion_config = load_diffusion_config(model)
        self.unet = load_unet(model, float16)
        self.text_encoder = load_text_encoder(model, float16)
        self.autoencoder = load_autoencoder(model, float16)
        self.sampler = SimpleEulerSampler(self.diffusion_config)
        self.tokenizer = load_tokenizer(model)

    def _get_text_conditioning(self, text: str, n_images: int = 1, cfg_weight: float = 7.5, negative_text: str = ""):
        """
        Function to get text conditioning for the model.
        """
        tokens = [self.tokenizer.tokenize(text)]
        if cfg_weight > 1:
            tokens += [self.tokenizer.tokenize(negative_text)]
        tokens = [t + [0] * (max(len(t) for t in tokens) - len(t)) for t in tokens]
        conditioning = self.text_encoder(mx.array(tokens))
        if n_images > 1:
            conditioning = _repeat(conditioning, n_images, axis=0)
        return conditioning

    def _denoising_step(self, x_t, t, t_prev, conditioning, cfg_weight: float = 7.5):
        """
        Function to perform a denoising step.
        """
        x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
        eps_pred = self.unet(x_t_unet, mx.broadcast_to(t, [len(x_t_unet)]), encoder_x=conditioning)
        if cfg_weight > 1:
            eps_text, eps_neg = eps_pred.split(2)
            eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)
        return self.sampler.step(eps_pred, x_t, t, t_prev)

    def _denoising_loop(self, x_T, T, conditioning, num_steps: int = 50, cfg_weight: float = 7.5, latent_image_placeholder=None):
        """
        Function to perform the denoising loop.
        """
        x_t = x_T
        for t, t_prev in self.sampler.timesteps(num_steps, start_time=T, dtype=self.dtype):
            x_t = self._denoising_step(x_t, t, t_prev, conditioning, cfg_weight)
            visualize_tensor(x_t, normalize=True, placeholder=latent_image_placeholder)
            yield x_t

    def generate_latents(self, text: str, input_image: Image.Image = None, n_images: int = 1, num_steps: int = 50, cfg_weight: float = 7.5, negative_text: str = "", latent_size: Tuple[int] = (64, 64), seed=-1, denoising_strength=0.7):
        """
        Function to generate latents.
        """
        mx.random.seed(int(seed))
        debug_print(f"Seed: {seed}")
        conditioning = self._get_text_conditioning(text, n_images, cfg_weight, negative_text)
        if input_image is not None:
            start_step = self.sampler.max_time * denoising_strength
            num_steps = int(num_steps * denoising_strength)
            input_image = normalize_tensor(mx.array(np.array(input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)))), (0, 255), (-1, 1))
            x_0, _ = self.autoencoder.encode(input_image[None])
            x_t = self.sampler.add_noise(mx.broadcast_to(x_0, [n_images] + list(x_0.shape[1:])), mx.array(start_step))
        else:
            x_t = self.sampler.sample_prior((n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype)

        # Visualize the latent space
        st.text("Starting Latent Space")
        visualize_tensor(x_t, normalize=True)

        st.text("Denoising Latent Space")
        latent_image_placeholder = st.empty()
        yield from self._denoising_loop(x_t, start_step if input_image is not None else self.sampler.max_time, conditioning, num_steps, cfg_weight, latent_image_placeholder)

    def decode(self, x_t):
        """
        Function to decode the latents.
        """
        x = self.autoencoder.decode(x_t)
        return mx.minimum(1, mx.maximum(0, x / 2 + 0.5))



