"""Join COCO image annotations with explicit object mentions in captions."""
import json
import re
import zipfile
from collections import Counter, defaultdict


# Caption vocabulary; names in the output always come from COCO categories.
ALIASES = {
    "person": r"people|persons?|men|man|women|woman|boys?|girls?|children|child|bab(?:y|ies)|toddlers?",
    "bicycle": r"bicycles?|pedal bikes?",
    "motorcycle": r"motorcycles?|motorbikes?",
    "car": r"cars?(?!\s+(?:dealerships?|keys?|wash|seats?)\b)",
    "bus": r"bus(?:es)?(?!\s+(?:stop|stations?)\b)",
    "train": r"trains?(?!\s+(?:stations?|tracks?|yards?)\b)",
    "umbrella": r"(?<!cocktail )(?<!toy )umbrellas?",
    "airplane": r"air[ -]?planes?|aeroplanes?|planes?|(?:jet|passenger) aircraft",
    "traffic light": r"traffic (?:lights?|signals?)|stoplights?|stop lights?",
    "fire hydrant": r"(?:fire )?hydrants?",
    "bird": r"birds?|seagulls?|pigeons?|ducks?|geese|goose|parrots?",
    "dog": r"(?<!hot )(?<!hot-)dogs?|pupp(?:y|ies)",
    "bear": r"(?<!teddy )(?<!teddy-)bears?(?!\s+(?:cub )?statue)",
    "cow": r"cows?|cattle|bulls?|calves|calf",
    "handbag": r"handbags?|purses?",
    "backpack": r"backpacks?|back packs?|rucksacks?",
    "tie": r"neckties?|neck ties?|bow ties?|(?:wearing|wears|with) (?:a |his |her )?(?:\w+ ){0,2}tie",
    "skis": r"skis|ski poles and skis",
    "sports ball": r"(?:soccer|tennis|baseball|basketball|volleyball|football|softball|golf|bowling|beach|rugby) balls?|basketballs?|footballs?|volleyballs?|soccerballs?",
    "baseball glove": r"baseball (?:gloves?|mitts?)|(?:catcher.s|catching) mitts?",
    "tennis racket": r"tennis (?:racquets?|rackets?)",
    "knife": r"knife|knives",
    "orange": r"oranges(?!\s+and yellows)|(?:peeled|sliced|whole|an? cut) orange|orange (?:slices?|segments?)",
    "hot dog": r"hot[ -]?dogs?",
    "donut": r"doughnuts?|donuts?",
    "couch": r"couch(?:es)?|sofas?|loveseats?",
    "potted plant": r"potted (?:plants?|flowers?|trees?)|plants? in (?:\w+ ){0,3}pots?|(?:flower|plant) ?pots? (?:with|containing) (?:\w+ ){0,2}(?:plants?|flowers?)",
    "dining table": r"(?:dining(?: room)?|kitchen|dinner) tables?|tables? (?:set for|set with|covered with) (?:\w+ ){0,2}(?:dinner|lunch|plates|dishes)",
    "tv": r"tvs?|televisions?|television sets?",
    "mouse": r"(?:computer|wireless|optical) mouse|(?:keyboard|computer|laptop) and (?:a )?mouse|mouse and (?:a )?(?:keyboard|computer|laptop)",
    "remote": r"remotes? controls?|tv remotes?|(?:holding|with) (?:a |the )?remote",
    "keyboard": r"keyboards?",
    "cell phone": r"cell[ -]?phones?|mobile phones?|smart ?phones?|iphones?",
    "microwave": r"microwaves?(?: ovens?)?",
    "oven": r"(?<!microwave )(?<!toaster )ovens?",
    "toaster": r"toasters?(?!\s+ovens?)",
    "refrigerator": r"refrigerators?|fridges?",
    "teddy bear": r"teddy[ -]?bears?|stuffed (?:toy )?bears?",
    "hair drier": r"hair[ -]?dr[yi]ers?|blow[ -]?dr[yi]ers?",
    "toothbrush": r"tooth[ -]?brush(?:es)?",
    "scissors": r"scissors|shears",
}


def class_matchers(names):
    return {
        name: re.compile(r"\b(?:" + ALIASES.get(name, re.escape(name) + r"(?:es|s)?") + r")\b", re.I)
        for name in names
    }


# Reject negated or incidental mentions, for example "a bicycle shop"
# or "books about motorcycles". Require the category to be annotated too.
INDIRECT_PREFIX = re.compile(
    r"\b(?:no|without|about|for attaching|shaped like|to look like|"
    r"(?:photos?|pictures?|sculptures?|models?|drawings?|paintings?) of)(?: \w+){0,2} $",
    re.I,
)
INDIRECT_SUFFIX = re.compile(
    r"^[- ](?:shops?|stores?|tracks?|racks?|teams?|themed|designs?|logos?|food|made)\b", re.I
)
DECORATION_PREFIX = re.compile(
    r"\b(?:statue|lego figures?|toy figures?|cartoon figures?)\b[^.!?;]*"
    r"\b(?:holding|depicting)\b[^.!?;]*$|"
    r"\bbirthday cake\b[^.!?;]*\b(?:with|features?|fashioned to look like)\b[^.!?;]*$",
    re.I,
)


def caption_classes(caption, image_classes, matchers):
    caption = " ".join(caption.split())
    return [
        name for name in image_classes
        if any(not INDIRECT_PREFIX.search(caption[:m.start()])
               and not DECORATION_PREFIX.search(caption[:m.start()])
               and not INDIRECT_SUFFIX.match(caption[m.end():])
               for m in matchers[name].finditer(caption))
    ]


def load_coco(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("annotations/instances_val2014.json") as source:
            instances = json.load(source)
        with archive.open("annotations/captions_val2014.json") as source:
            captions = json.load(source)["annotations"]
    names = {row["id"]: row["name"] for row in instances["categories"]}
    image_classes = defaultdict(set)
    for row in instances["annotations"]:
        image_classes[row["image_id"]].add(names[row["category_id"]])
    counts = Counter(name for labels in image_classes.values() for name in labels)
    counts = {name: counts[name] for name in sorted(names.values())}
    return captions, image_classes, counts


def preferred_captions(captions, image_classes, counts):
    """Keep one real caption per image, preferring explicit rarer objects."""
    matchers = class_matchers(counts)
    selected = {}
    for row in captions:
        image_id = row["image_id"]
        labels = sorted(image_classes[image_id])
        described = caption_classes(row["caption"], labels, matchers)
        score = sum(1 / counts[name] for name in described)
        if image_id not in selected or score > selected[image_id][0]:
            selected[image_id] = (score, row, labels, described)
    return ((row, labels, described) for _, row, labels, described in selected.values())
