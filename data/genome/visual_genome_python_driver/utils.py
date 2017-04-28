from models import Image, Object, Attribute, Relationship
from models import Region, Graph, QA, QAObject, Synset
import httplib
import json

"""
Get the local directory where the Visual Genome data is locally stored.
"""
def GetDataDir():
  from os.path import dirname, realpath, join
  dataDir = join(dirname(realpath('__file__')), 'data')
  return dataDir

"""
Helper Method used to get all data from request string.
"""
def RetrieveData(request):
  connection = httplib.HTTPConnection("visualgenome.org", '443')
  connection.request("GET", request)
  response = connection.getresponse()
  jsonString = response.read()
  data = json.loads(jsonString)
  return data

"""
Helper to Extract Synset from canon object.
"""
def ParseSynset(canon):
  if len(canon) == 0:
    return None
  return Synset(canon[0]['synset_name'], canon[0]['synset_definition'])

"""
Helper to parse a Graph object from API data.
"""
def ParseGraph(data, image):
  objects = []
  object_map = {}
  relationships = []
  attributes = []
  # Create the Objects
  for obj in data['bounding_boxes']:
    names = []
    synsets = []
    for s in obj['boxed_objects']:
      names.append(s['name'])
      synsets.append(ParseSynset(s['object_canon']))
      object_ = Object(obj['id'], obj['x'], obj['y'], obj['width'], obj['height'], names, synsets)
      object_map[obj['id']] = object_
    objects.append(object_)
  # Create the Relationships
  for rel in data['relationships']:
    relationships.append(Relationship(rel['id'], object_map[rel['subject']], \
        rel['predicate'], object_map[rel['object']], ParseSynset(rel['relationship_canon'])))
  # Create the Attributes
  for atr in data['attributes']:
    attributes.append(Attribute(atr['id'], object_map[atr['subject']], \
        atr['attribute'], ParseSynset(atr['attribute_canon'])))
  return Graph(image, objects, relationships, attributes)

"""
Helper to parse the image data for one image.
"""
def ParseImageData(data):
  img_id = data['id'] if 'id' in data else data['image_id']
  url = data['url']
  width = data['width']
  height = data['height']
  coco_id = data['coco_id']
  flickr_id = data['flickr_id']
  image = Image(img_id, url, width, height, coco_id, flickr_id)
  return image

"""
Helper to parse region descriptions.
"""
def ParseRegionDescriptions(data, image):
  regions = []
  if data[0].has_key('region_id'):
    region_id_key = 'region_id'
  else:
    region_id_key = 'id'
  for d in data:
    regions.append(Region(d[region_id_key], image, d['phrase'], d['x'], d['y'], d['width'], d['height']))
  return regions

"""
Helper to parse a list of question answers.
"""
def ParseQA(data, image_map):
  qas = []
  for d in data:
    qos = []
    aos = []
    if 'question_objects' in d:
      for qo in d['question_objects']:
        synset = Synset(qo['synset_name'], qo['synset_definition'])
        qos.append(QAObject(qo['entity_idx_start'], qo['entity_idx_end'], qo['entity_name'], synset))
    if 'answer_objects' in d:
      for ao in d['answer_objects']:
        synset = Synset(o['synset_name'], ao['synset_definition'])
        aos.append(QAObject(ao['entity_idx_start'], ao['entity_idx_end'], ao['entity_name'], synset))
    qas.append(QA(d['qa_id'], image_map[d['image_id']], d['question'], d['answer'], qos, aos))
  return qas
