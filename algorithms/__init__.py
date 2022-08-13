from algorithms.vdn import Vdn
from algorithms.qmix import Qmix
from algorithms.qatten import Qatten
from algorithms.qtran import Qtran
from algorithms.coma import Coma


REGISTRY = {}
REGISTRY['vdn'] = Vdn
REGISTRY['qmix'] = Qmix
REGISTRY['qatten'] = Qatten
REGISTRY['qtran'] = Qtran
REGISTRY['coma'] = Coma
