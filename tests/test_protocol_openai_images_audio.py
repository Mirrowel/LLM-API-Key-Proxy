from __future__ import annotations

from rotator_library.protocols import (
    OPERATION_AUDIO_TRANSCRIPTION,
    OPERATION_IMAGE_EDIT,
    OPERATION_IMAGE_GENERATION,
    OPERATION_SPEECH,
    ProtocolContext,
    get_protocol,
)


def test_openai_images_generation_and_edit_shapes() -> None:
    adapter = get_protocol("openai_images")
    generation = adapter.parse_request({"model": "gpt-image-test", "prompt": "draw a red cube", "size": "1024x1024"})
    edit = adapter.parse_request({"model": "gpt-image-test", "prompt": "make it blue", "image": "image-ref", "mask": "mask-ref"})
    response = adapter.parse_response({"data": [{"url": "https://example.test/image.png", "revised_prompt": "cube"}]})
    edit_response = adapter.parse_response(
        {"data": [{"url": "https://example.test/edit.png"}]},
        ProtocolContext(provider_options={"operation": OPERATION_IMAGE_EDIT}),
    )

    assert generation.operation == OPERATION_IMAGE_GENERATION
    assert adapter.build_request(generation)["prompt"] == "draw a red cube"
    assert edit.operation == OPERATION_IMAGE_EDIT
    assert {entry["field"] for entry in edit.files} == {"image", "mask"}
    assert adapter.build_request(edit)["image"] == "image-ref"
    assert response.data[0]["revised_prompt"] == "cube"
    assert edit_response.operation == OPERATION_IMAGE_EDIT


def test_openai_audio_transcription_and_speech_shapes() -> None:
    adapter = get_protocol("openai_audio")
    transcription = adapter.parse_request({"model": "whisper-test", "file": "audio-ref", "language": "en"})
    speech = adapter.parse_request({"model": "tts-test", "input": "hello", "voice": "alloy"})
    text_response = adapter.parse_response({"text": "hello world"})
    translation_response = adapter.parse_response(
        {"text": "bonjour"},
        ProtocolContext(provider_options={"operation": "audio_translation"}),
    )
    binary_response = adapter.parse_response(b"RIFF")

    assert transcription.operation == OPERATION_AUDIO_TRANSCRIPTION
    assert transcription.files[0]["value"] == "audio-ref"
    assert speech.operation == OPERATION_SPEECH
    assert adapter.build_request(speech)["voice"] == "alloy"
    assert text_response.output == ["hello world"]
    assert translation_response.operation == "audio_translation"
    assert binary_response.content_type == "application/octet-stream"
