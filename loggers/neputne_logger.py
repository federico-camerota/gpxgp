import neptune.new as neptune
import yaml


class NeptuneLogger:

    def __init__(self, credentials_file, parameters=None):
        with open(credentials_file, 'r') as f:
            try:
                credentials = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        self._run = neptune.init_run(credentials['project'], credentials['api_token'], source_files=[])

        if parameters is not None:
            self._run['parameters'] = parameters

    def log_entry(self, entry, value):
        self._run[entry] = value

    def log_metric(self, metric, value):
        self._run[metric].log(value)

    def upload(self, entry, value):
        self._run[entry].upload(value)

    def upload_files(self, entry, value):
        self._run[entry].upload_files(value)

    def stop(self):
        self._run.stop()