class VoiceAgentError(Exception):
    pass


class VoiceProviderError(VoiceAgentError):
    pass


class TranscriptionError(VoiceProviderError):
    pass


class TTSError(VoiceProviderError):
    pass


class LLMError(VoiceAgentError):
    pass


class SessionError(VoiceAgentError):
    pass


class ConfigurationError(VoiceAgentError):
    pass


class RetryableError(VoiceAgentError):
    pass
