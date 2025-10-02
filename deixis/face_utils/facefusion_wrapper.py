# facefusion_wrapper.py
from __future__ import annotations
from typing import Iterable
from argparse import ArgumentParser, HelpFormatter
import torch

import facefusion.choices
from facefusion import config, metadata, state_manager, wording
from facefusion.common_helper import create_float_metavar, create_int_metavar, get_first, get_last
from facefusion.execution import get_available_execution_providers
from facefusion.ffmpeg import get_available_encoder_set
from facefusion.filesystem import get_file_name, resolve_file_paths
from facefusion.jobs import job_store
from facefusion.processors.core import get_processors_modules

from facefusion import face_classifier, face_detector, face_recognizer, state_manager, wording
from facefusion.filesystem import get_file_name, resolve_file_paths
from facefusion.processors.core import get_processors_modules

from facefusion import face_detector, face_classifier, face_recognizer
from facefusion.program_helper import validate_args
from facefusion.core import route
from facefusion.args import apply_args
from facefusion.program import create_config_path_program, create_temp_path_program, create_jobs_path_program, create_source_paths_program, create_target_path_program, create_output_path_program, collect_step_program, create_uis_program, create_benchmark_program, collect_job_program, create_download_providers_program, create_download_scope_program, create_log_level_program, create_job_status_program, create_job_id_program, create_step_index_program, create_halt_on_error_program, create_source_pattern_program, create_target_pattern_program, create_output_pattern_program

# Optional: only if you want true 68-point landmarking (not the 5→68 estimate)
try:
    from facefusion import face_landmarker  # may not exist in some builds
    _HAS_LANDMARKER = True
except Exception:
    _HAS_LANDMARKER = False


def create_help_formatter_small(prog : str) -> HelpFormatter:
	return HelpFormatter(prog, max_help_position = 50)


def create_help_formatter_large(prog : str) -> HelpFormatter:
	return HelpFormatter(prog, max_help_position = 300)

def update_download_providers():
	common_modules =\
	[
		face_classifier,
		face_detector,
		face_recognizer,
	]

	for module in common_modules:
		if hasattr(module, 'create_static_model_set'):
			module.create_static_model_set.cache_clear()

	download_providers = facefusion.choices.download_providers
	state_manager.set_item('download_providers', download_providers)
	return 


def create_program() -> ArgumentParser:
	program = ArgumentParser(formatter_class = create_help_formatter_large, add_help = False)
	# program._positionals.title = 'commands'
	# program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
	# sub_program = program.add_subparsers(dest = 'command')
	# # general
	# sub_program.add_parser('run', help = wording.get('help.run'), parents = [ create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), create_source_paths_program(), create_target_path_program(), create_output_path_program(), collect_step_program(), create_uis_program(), create_benchmark_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('headless-run', help = wording.get('help.headless_run'), parents = [ create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), create_source_paths_program(), create_target_path_program(), create_output_path_program(), collect_step_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('batch-run', help = wording.get('help.batch_run'), parents = [ create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), create_source_pattern_program(), create_target_pattern_program(), create_output_pattern_program(), collect_step_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('force-download', help = wording.get('help.force_download'), parents = [ create_download_providers_program(), create_download_scope_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('benchmark', help = wording.get('help.benchmark'), parents = [ create_temp_path_program(), collect_step_program(), create_benchmark_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# # job manager
	# sub_program.add_parser('job-list', help = wording.get('help.job_list'), parents = [ create_job_status_program(), create_jobs_path_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-create', help = wording.get('help.job_create'), parents = [ create_job_id_program(), create_jobs_path_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-submit', help = wording.get('help.job_submit'), parents = [ create_job_id_program(), create_jobs_path_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-submit-all', help = wording.get('help.job_submit_all'), parents = [ create_jobs_path_program(), create_log_level_program(), create_halt_on_error_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-delete', help = wording.get('help.job_delete'), parents = [ create_job_id_program(), create_jobs_path_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-delete-all', help = wording.get('help.job_delete_all'), parents = [ create_jobs_path_program(), create_log_level_program(), create_halt_on_error_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-add-step', help = wording.get('help.job_add_step'), parents = [ create_job_id_program(), create_config_path_program(), create_jobs_path_program(), create_source_paths_program(), create_target_path_program(), create_output_path_program(), collect_step_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-remix-step', help = wording.get('help.job_remix_step'), parents = [ create_job_id_program(), create_step_index_program(), create_config_path_program(), create_jobs_path_program(), create_source_paths_program(), create_output_path_program(), collect_step_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-insert-step', help = wording.get('help.job_insert_step'), parents = [ create_job_id_program(), create_step_index_program(), create_config_path_program(), create_jobs_path_program(), create_source_paths_program(), create_target_path_program(), create_output_path_program(), collect_step_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-remove-step', help = wording.get('help.job_remove_step'), parents = [ create_job_id_program(), create_step_index_program(), create_jobs_path_program(), create_log_level_program() ], formatter_class = create_help_formatter_large)
	# # job runner
	# sub_program.add_parser('job-run', help = wording.get('help.job_run'), parents = [ create_job_id_program(), create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-run-all', help = wording.get('help.job_run_all'), parents = [ create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), collect_job_program(), create_halt_on_error_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-retry', help = wording.get('help.job_retry'), parents = [ create_job_id_program(), create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), collect_job_program() ], formatter_class = create_help_formatter_large)
	# sub_program.add_parser('job-retry-all', help = wording.get('help.job_retry_all'), parents = [ create_config_path_program(), create_temp_path_program(), create_jobs_path_program(), collect_job_program(), create_halt_on_error_program() ], formatter_class = create_help_formatter_large)
	return ArgumentParser(parents = [ program ], formatter_class = create_help_formatter_small)

def init_facefusion_state(
    *,
    detector_model: str = "retinaface",          # "retinaface" | "scrfd" | "yolo_face" | "yunet" | "many"
    detector_size: str = "640x640",              # must be WxH, divisible by 32 is safe
    detector_score: float = 0.6,                 # detection threshold
    detector_angles: Iterable[int] = (0,),       # e.g. (0, 90, -90)
    use_landmarker_68: bool = False,            # enable true 68-pt landmarker
    landmarker_score: float = 0.5,              # confidence threshold for 68-pt (if enabled)
    download_scope: str = "full",               # for model downloads
) -> None:
    """
    Initialize the keys the CLI would usually set so that detectors/recognizers run headlessly.
    Safe defaults mirror a typical FaceFusion run.
    """
    update_download_providers()
    program = create_program()
    # print(program)
    # print(program.parse_args())
    # return
    # if validate_args(program):
    #     args = vars(program.parse_args())
    #     print(args)
    #     apply_args(args, state_manager.init_item)

    # Set in both 'cli' and 'ui' states
    state_manager.init_item("download_scope", download_scope)

    # Detector config used by face_detector.detect_faces()
    state_manager.init_item("face_detector_model", detector_model)
    state_manager.init_item("face_detector_size", detector_size)
    state_manager.init_item("face_detector_score", float(detector_score))
    state_manager.init_item("face_detector_angles", list(detector_angles))

    # Landmarker gate used by create_faces(); when 0, falls back to 5→68 estimate
    state_manager.init_item("face_landmarker_score", float(landmarker_score if use_landmarker_68 else 0.0))
    state_manager.init_item("face_landmarker_model", 'many')


    # # execution
    state_manager.init_item("execution_device_ids", ['0'])
    state_manager.init_item("execution_providers", ["cuda" if torch.cuda.is_available() else "coreml" if torch.backends.mps.is_available() else "cpu"])
    # state_manager.init_item("execution_thread_count", execution_thread_count)
    # download




def ensure_facefusion_models(use_landmarker_68: bool = False) -> None:
    """
    Download (if needed) and prepare inference pools for the modules used by get_many_faces().
    Must be called once after init_facefusion_state().
    """
    # These pre_check() calls perform conditional model downloads
    ok = True
    ok &= bool(face_detector.pre_check())
    ok &= bool(face_classifier.pre_check())   # needed because create_faces() always calls classify_face()
    ok &= bool(face_recognizer.pre_check())   # ArcFace for embeddings

    if use_landmarker_68 and _HAS_LANDMARKER:
        ok &= bool(face_landmarker.pre_check())

    if not ok:
        raise RuntimeError("FaceFusion model pre_check() failed; see logs for which module failed.")

    # Build inference pools now (avoid first-call latency later)
    face_detector.get_inference_pool()
    face_classifier.get_inference_pool()
    face_recognizer.get_inference_pool()
    if use_landmarker_68 and _HAS_LANDMARKER:
        face_landmarker.get_inference_pool()


def shutdown_facefusion() -> None:
    """
    Optional: free ONNXRuntime sessions when you’re done (useful for batch workers).
    """
    try:
        face_detector.clear_inference_pool()
    except Exception:
        pass
    try:
        face_classifier.clear_inference_pool()
    except Exception:
        pass
    try:
        face_recognizer.clear_inference_pool()
    except Exception:
        pass
    if _HAS_LANDMARKER:
        try:
            face_landmarker.clear_inference_pool()
        except Exception:
            pass