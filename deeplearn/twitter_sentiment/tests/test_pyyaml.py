import yaml
import pprint

foo = yaml.load(open('test2.yaml').read())
pprint.pprint(foo)
