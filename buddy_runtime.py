""" The highest abstraction level of buddy.py, a minimal and modular framework
for an LLM-based AI buddy.

Goals
- model stream-of-consciousness, not chat log
- assume simple RAG
- has idle loop
- has context-swap and focus mode
- modular

Elements
- _Event: external input (e.g. a message)
- _Action: self-output (e.g. a thought) or external output (e.g. a reply)
- _Memory: an entry of memory that would fit in the RAG DB
- _Cogneme: an entry on the buddy's context
- _State: a subaction state, in case an action takes multiple prompts
- Frame: a list of cognemes under a purpose or theme

Abstractions
- Responder: wraps the LLM backend.
- Indexer: manages long-term memory.
- Periphery: encodes events, dispatches actions.
- Guide: manages sub-action state machine.
- Context: tracks context (list of frames).
- Runtime: coordinator of the above.
"""

from __future__ import annotations
from typing import (
    Callable as Fn,
    Generic as Of,
    Iterator,
    Protocol as Sig,
    Sequence as Seq,
    TypeAlias as Typ,
    TypeVar as Tyvar,
)
from dataclasses import dataclass
from logging import Logger

import time


SuccessBit: Typ = bool
TagStr: Typ = str


_T = Tyvar('_T', covariant=True)

_Event = Tyvar('_Event')
_Action = Tyvar('_Action')
_Action_Co = Tyvar('_Action_Co', covariant=True)
_Memory = Tyvar('_Memory')
_Cogneme = Tyvar('_Cogneme')
_State = Tyvar('_State')


@dataclass
class Frame(Of[_Cogneme]):
    summary: _Cogneme
    details: list[_Cogneme]

class Streamed(Of[_T], Sig):
    def __next__(self) -> _T: ...
    def close(self) -> None: ...
    def is_closed(self) -> bool: ...

Response: Typ = Streamed[str]
Responder: Typ = Fn[[str], Response]


class Indexer(Of[_Memory, _Cogneme], Sig):
    def pack_memory(self, memory: _Memory) -> _Cogneme: ...
    def unpack_memory(self, cogneme: _Cogneme) -> _Memory | None: ...

    def search(
            self,
            terms: Seq[_Cogneme],
            require: Seq[TagStr],
            exclude: Seq[TagStr],
            first: int,
            *,
            bump: bool,
            ) -> list[_Memory]:
        ...
    def update(
            self,
            item: _Memory,
            append: Seq[TagStr],
            remove: Seq[TagStr],
            *,
            bump: bool,
            ) -> SuccessBit:
        ...
    def save(self) -> SuccessBit: ...

class Periphery(Of[_Event, _Action, _Cogneme], Sig):
    def pack_event(self, event: _Event) -> _Cogneme: ...
    def unpack_event(self, cogneme: _Cogneme) -> _Event | None: ...
    def interrupt_level(self, event: _Event) -> float: ...

    def pack_action(self, action: _Action) -> _Cogneme: ...
    def unpack_action(self, cogneme: _Cogneme) -> _Action | None: ...

    def poll(self) -> list[_Event]: ...
    def actuate(self, action: _Action) -> None: ...

class Guide(Of[_Action_Co, _Cogneme, _State], Sig):
    def initial(self) -> _State: ...
    def is_relaxed(self, state: _State) -> bool: ...
    def heat_level(self, state: _State) -> float: ...
    def antiinterrupt_level(self, state: _State) -> float: ...
    def split(self, state: _State) -> tuple[_Action_Co, _State] | None: ...
    def prompt(self, state: _State, *frames: Frame[_Cogneme]) -> str: ...
    def parse(self, state: _State, res: Response) -> _State | None: ...

class Context(Of[_Event, _Action, _Memory, _Cogneme], Sig):
    def prepare_frames(
            self,
            indexer: Indexer[_Memory, _Cogneme],
            periphery: Periphery[_Event, _Action, _Cogneme],
            ) -> list[Frame[_Cogneme]]:
        ...
    def push_event(self, event: _Event) -> None: ...
    def internal_dispatch(self, action: _Action) -> SuccessBit: ...
    def save(self) -> None: ...


class Timer:
    delay_sec: float
    instant: float
    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = delay_sec
        self.instant = -1
    def set(self) -> Timer:
        self.instant = time.perf_counter() + self.delay_sec
        return self
    def clear(self) -> Timer:
        self.instant = -1
        return self
    def is_up(self) -> bool:
        return self.instant <= time.perf_counter()


@dataclass
class Runtime(Of[_Event, _Action, _Memory, _Cogneme, _State]):
    responder: Responder
    indexer: Indexer[_Memory, _Cogneme]
    periphery: Periphery[_Event, _Action, _Cogneme]
    guide: Guide[_Action, _Cogneme, _State]
    context: Context[_Event, _Action, _Memory, _Cogneme]

    def cognition_loop(
            self,
            *,
            base_delay_sec: float,
            idle_heat_boundary: float,
            idle_delay_sec: float,
            periodic_save_sec: float,
            nonsense_retries: int,
            nonsense_retry_delay_sec: float,
            event_trickle: bool,
            check_stop: Fn[[], bool],
            logger: Logger,
            ) -> None:
        """
        Okay I should really explain what the f- is going on.

        If our buddy is in the middle of formulating an action that takes
        multiple prompts, such as taking a mental note, we don't want to stop
        that. This is why the Guide decides whether a State is relaxed or not.
        When the composition is finished and Actions are ready for dispatch,
        the State becomes relaxed. A relaxed State has two characteristic
        values: heat level, and anti-interrupt level.

        Heat directly determines whether our buddy goes into idle mode: if a
        State is sufficiently hot, the loop continues immediately; otherwise,
        the idle delay is applied.

        During execution, there'll be Events, like discord pings. Each event
        has an interrupt level. If our buddy is amidst something serious,
        they'll have a sky-high anti-interrupt level, and the Events will be
        queued without even a hint of their presence. Otherwise, any queued
        Events with sufficiently-high interrupt levels are added to the
        context in chronological order, and will break any idle state.

        One can configure Events to trickle, so that they're dealt with
        one-by-one, instead of flooding the context. This is quite primitive,
        but it'll have to do for now.

        When a sub-action step is actually being taken, we check if the Context
        has been updated, and if so, refresh the frames of Cognemes; then we
        feed this to the Guide to get a prompt, feed the prompt to the LLM
        backend for a stream of str snippets, then pass the stream along with
        the current State to the Guide's parser.
        - If successful, we get a new State, which is then checked for Actions,
          which are then passed onto the Context to attempt internal
          resolution, and if deemed unresolved, passed onto the actuator.
        - If the parsing fails, this means either our parser's broken or our
          LLM generated some nonsense: this attempt is discarded, a retry timer
          is set, and after a set amount of retries, the loop will log the
          prompt and abort.
        """
        state = self.guide.initial()
        check_idle = lambda s: self.guide.is_relaxed(s) and \
                self.guide.heat_level(s) < idle_heat_boundary
        idleness_cache = check_idle(state)

        t_idle = Timer(idle_delay_sec)
        t_save = Timer(periodic_save_sec).set()
        t_retry = Timer(nonsense_retry_delay_sec)
        retries_left = nonsense_retries

        make_frames = lambda: self.context.prepare_frames(
                self.indexer, self.periphery )
        current_frames = make_frames()
        frames_need_refresh = False
        event_buffer = []

        while not check_stop():

            if self.guide.is_relaxed(state) and \
                    retries_left == nonsense_retries:
                I = self.guide.antiinterrupt_level(state)
                event_buffer.extend(self.periphery.poll())
                to_delete = []
                for j, event in enumerate(event_buffer):
                    i = self.periphery.interrupt_level(event)
                    if i > I:
                        self.context.push_event(event)
                        to_delete.append(j)
                        t_idle.clear()
                        frames_need_refresh = True
                        if event_trickle: break
                while len(to_delete) > 0:
                    event_buffer.remove(to_delete.pop())

            if t_save.is_up():
                if not self.indexer.save():
                    logger.warning(f"Couldn't save indexer {self.indexer}")
                if not self.context.save():
                    logger.warning(f"Couldn't save context {self.context}")

            if t_retry.is_up() and (not idleness_cache or t_idle.is_up()):
                if frames_need_refresh:
                    current_frames = make_frames()
                    frames_need_refresh = False
                prompt = self.guide.prompt(state, *current_frames)
                response = self.responder(prompt)
                new_state = self.guide.parse(state, response)
                if new_state is None:
                    t_retry.set()
                    t_idle.clear()
                    retries_left -= 1
                    message = (
                        f"({retries_left=}) "
                        f"Cannot satisfy the parser at {prompt=}"
                    )
                    if retries_left <= 0:
                        logger.critical(message)
                        raise RuntimeError(message)
                    else:
                        logger.warning(message)
                else:
                    t_retry.clear()
                    t_idle.set()
                    retries_left = nonsense_retries
                    state = new_state
                    while (split := self.guide.split(state)) is not None:
                        action, state = split
                        if self.context.internal_dispatch(action):
                            frames_need_refresh = True
                        else:
                            self.periphery.actuate(action)
                    idleness_cache = check_idle(state)
            else:
                time.sleep(base_delay_sec)

        if not self.indexer.save():
            logger.error(f"Couldn't save indexer {self.indexer} on exit")
        if not self.context.save():
            logger.error(f"Couldn't save context {self.context} on exit")
