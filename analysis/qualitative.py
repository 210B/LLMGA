'''
from a2c_qda_pipeline_lib import run_full_pipeline
run_full_pipeline(
        hamlet_path="data/questionnaire/text_hamlet.csv",
        venice_path="data/questionnaire/text_venice.csv",
        model="gpt-4.1",
        batch_size=25,
        dotenv_path=".env",                 # optional
        outdir="outputs/my_run"            # optional; default: outputs/<timestamp>
    )
'''
from a2c_qda_pipeline_lib_min import run_full_pipeline
run_full_pipeline("data/questionnaire/text_venice.csv", "data/questionnaire/text_hamlet.csv",
                  model="gpt-4.1",
                  dotenv_path=".env",
                  encoding="utf-8-sig",
                  group_map_path="data/questionnaire/실험참가자최종.xlsx",
                  group_map_id_col="ID",
                  group_map_group_col="모델 구분",
                  by_group_summary=True)