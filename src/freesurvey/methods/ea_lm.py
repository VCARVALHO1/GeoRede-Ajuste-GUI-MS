"""EA-LM: Euler-angle Levenberg-Marquardt adjustment."""

from freesurvey.Registration_Module import Method4_GlobalEulerLM


class EA_LM(Method4_GlobalEulerLM):
    method_code = "EA-LM"
    article_name = "Euler-angle Levenberg-Marquardt global similarity adjustment"


Method4_GlobalEulerLM_Alias = EA_LM
