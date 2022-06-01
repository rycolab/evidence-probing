from typing import List, Union, Dict, Any
import torch


class ProbingUnit:
    """
    (Abstract) probing units, from which concretely probably units (e.g., subtokens, words, arcs, etc.)
    should derive from.
    """
    def __init__(self, embedding: torch.Tensor, attributes: Dict[str, str]):
        self._embedding = embedding
        self._attributes = attributes

    def get_embedding(self) -> torch.Tensor:
        return self._embedding

    def set_embedding(self, embedding: torch.Tensor) -> None:
        self._embedding = embedding

    def has_attribute(self, attr) -> bool:
        return attr in self._attributes

    def get_attribute(self, attr) -> Any:
        return self._attributes[attr]

    def get_attributes(self) -> List[str]:
        return list(self._attributes.keys())

    def __repr__(self) -> str:
        return f"({self._attributes})"

    def randomize(self) -> None:
        self._embedding = torch.randn_like(self._embedding)


class Word(ProbingUnit):
    def __init__(self, word: str, embedding: torch.Tensor, attributes: Dict[str, str]):
        self._word = word

        super().__init__(embedding=embedding, attributes=attributes)

    def get_word(self) -> str:
        return self._word

    def __repr__(self) -> str:
        return f"{self._word}({self._attributes})"


class Arc(ProbingUnit):
    def __init__(self, head: str, tail: str, embedding: torch.Tensor, attributes: Dict[str, str]):
        self._head = head
        self._tail = tail
        self.split = int(len(embedding) / 2)

        super().__init__(embedding=embedding, attributes=attributes)

    def get_head(self) -> str:
        return self._head

    def get_tail(self) -> str:
        return self._tail

    def __repr__(self) -> str:
        return f"{self._head}→{self._tail}({self._attributes})"

    def set_head_embedding(self, embedding: torch.Tensor) -> None:
        self._embedding[:self.split] = embedding

    def set_tail_embedding(self, embedding: torch.Tensor) -> None:
        self._embedding[self.split:] = embedding


class Inference(ProbingUnit):
    def __init__(self, premise: str, hypothesis: str, embedding: torch.Tensor, attributes: Dict[str, str]):
        self._premise = premise
        self._hypothesis = hypothesis

        super().__init__(embedding=embedding, attributes=attributes)

    def get_premise(self) -> str:
        return self._premise

    def get_hypothesis(self) -> str:
        return self._hypothesis

    def __repr__(self) -> str:
        return f"{self._premise}=>{self._hypothesis}({self._attributes})"


class BooleanQuestion(ProbingUnit):
    def __init__(self, question: str, passage: str, embedding: torch.Tensor, attributes: Dict[str, str]):
        self._question = question
        self._passage = passage

        super().__init__(embedding=embedding, attributes=attributes)

    def get_question(self) -> str:
        return self._question

    def get_passage(self) -> str:
        return self._passage

    def __repr__(self) -> str:
        return f"[{self._passage[:40]}...]::{self._question}?({self._attributes})"


class COPAQuestion(ProbingUnit):
    def __init__(self, premise: str, choice1: str, choice2: str,
                 embedding: torch.Tensor, attributes: Dict[str, str]):
        self._premise = premise
        self._choice1 = choice1
        self._choice2 = choice2

        super().__init__(embedding=embedding, attributes=attributes)

    def get_premise(self) -> str:
        return self._premise

    def get_choice1(self) -> str:
        return self._choice1

    def get_choice2(self) -> str:
        return self._choice2

    def __repr__(self) -> str:
        return f"[{self._premise[:40]}...]::{self._choice1}/{self._choice2}?({self._attributes})"


PyTorchDevice = Union[torch.device, str]
Specification = Dict[str, Any]
PropertyValue = str
