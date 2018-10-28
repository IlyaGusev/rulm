import json

class NNConfig:
    def save(self, file_name):
        assert file_name.endswith(".json")
        with open(file_name, 'w', encoding="utf-8") as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + '\n')

    def load(self, file_name):
        assert file_name.endswith(".json")
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)

