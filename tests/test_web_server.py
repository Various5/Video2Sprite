from video2spritesheet.web.server import GenerationRequest


def test_generation_request_parses_colors_and_max_frames_zero_to_none():
    req = GenerationRequest.model_validate(
        {
            "background_color": "10,20,30,40",
            "chroma_key_color": "1,2,3",
            "max_frames": 0,
            "padding": 5,
        }
    )
    assert req.background_color == (10, 20, 30, 40)
    assert req.chroma_key_color == (1, 2, 3, 255)
    assert req.max_frames is None
