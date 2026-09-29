"""Repository maintenance: generate the simple independent local-package manifests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    ("shared/contracts", "medication-contracts", "medication_contracts", ['pydantic>=2.11,<3']),
    ("modules/authentication-tracking", "medication-authentication-tracking", "authentication_tracking", ['medication-contracts==0.1.0']),
    ("modules/event-detection", "medication-event-detection", "event_detection", ['medication-contracts==0.1.0']),
    ("modules/session-video-management", "medication-session-video", "session_video", ['medication-contracts==0.1.0', 'av>=14,<17', 'Pillow>=11,<13', 'numpy>=1.26,<3']),
    ("modules/session-decision", "medication-session-decision", "session_decision", ['medication-contracts==0.1.0', 'vlm-event-verification==0.1.0']),
    ("runtime", "medication-pipeline", "medication_pipeline", [
        'medication-contracts==0.1.0', 'medication-authentication-tracking==0.1.0',
        'medication-event-detection==0.1.0', 'medication-session-video==0.1.0',
        'medication-session-decision==0.1.0', 'vlm-event-verification==0.1.0']),
]


def main():
    import json
    import os
    local = {name: ROOT / location for location, name, _, _ in PACKAGES}
    local['vlm-event-verification'] = ROOT / 'modules/vlm-event-detection'
    for location, name, package, deps in PACKAGES:
        directory = ROOT / location
        content = ('[build-system]\nrequires = ["hatchling>=1.27,<2"]\n'
                   'build-backend = "hatchling.build"\n\n[project]\n'
                   f'name = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.10,<3.14"\n'
                   f'dependencies = {json.dumps(deps)}\n\n'
                   f'[tool.hatch.build.targets.wheel]\npackages = ["src/{package}"]\n')
        if name == 'medication-event-detection':
            content += '\n[project.optional-dependencies]\nmodels = ["ultralytics>=8.3,<9", "mediapipe>=0.10.21,<0.11", "numpy>=1.26,<3"]\n'
        if name == 'medication-pipeline':
            content += ('\n[project.optional-dependencies]\n'
                'models = ["medication-event-detection[models]==0.1.0", "vlm-event-verification[inference]==0.1.0"]\n'
                'detection = ["medication-event-detection[models]==0.1.0"]\n'
                'dev = ["pytest>=8,<10", "ruff>=0.11,<1"]\n'
                '\n[project.scripts]\nmedication-pipeline = "medication_pipeline.cli:main"\n')
        content += '\n[tool.uv.sources]\n'
        # uv resolves transitive path dependencies from the invoking project.
        for dependency, path in local.items():
            if dependency != name:
                relative = Path(os.path.relpath(path, directory)).as_posix()
                content += f'{dependency} = {{ path = "{relative}", editable = true }}\n'
        (directory / 'pyproject.toml').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
