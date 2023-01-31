from pathlib import Path
import yappi

from regressor_on_resnet.metadata_loader import MetadataLoader

this_script_path = Path(__file__)
result_dir_path = this_script_path.parent / 'results'
project_root_path = this_script_path.parent.parent.parent

yappi.set_clock_type("cpu")
yappi.start()

metadata_loader = MetadataLoader(
    (
        project_root_path / 'cloud_applications_v2/expeditions_configs/AI-58-config.json',
        project_root_path / 'cloud_applications_v2/expeditions_configs/AMK-79-config.json',
        project_root_path / 'cloud_applications_v2/expeditions_configs/ABP-42-config.json',
        project_root_path / 'cloud_applications_v2/expeditions_configs/AI-52-config.json',
        project_root_path / 'cloud_applications_v2/expeditions_configs/AI-49-config.json',
        # project_root_path / 'cloud_applications_v2/expeditions_configs/ANS-31-config.json',
    ),
    radiation_class_number=8)

yappi.stop()

result_dir_path.mkdir(exist_ok=True)
func_stats = yappi.get_func_stats().save(
    this_script_path.parent / 'results' / f'{this_script_path.name[:-3]}.pstat', type="pstat")
