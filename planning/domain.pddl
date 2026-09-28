(define (domain cifar100-process)
  (:requirements :strips :typing :negative-preconditions :equality)

  (:types
    item location - object
  )

  (:constants
    lab outdoors - location
    knife dslr - item
  )

  (:predicates
    ;; --- Location & Inventory ---
    (agent-at ?l - location)
    (at ?i - item ?l - location)
    (holding ?i - item)
    (hand-empty)

    ;; --- Stacking ---
    (on-top ?top - item ?bottom - item)
    (clear ?i - item)  ; Nothing is on top of this item

    ;; --- Static Properties ---
    (is-tool ?i - item)
    (is-surface ?i - item)  ; Can have things placed on it

    ;; --- Transformations ---
    (whole ?i - item)
    (cut-into-pieces ?i - item)
    (documented ?i - item)
  )

  ;; =========================================================================
  ;; MOVEMENT
  ;; =========================================================================

  (:action walk-between-rooms
    :parameters (?from - location ?to - location)
    :precondition (and
      (agent-at ?from)
      (not (= ?from ?to))
    )
    :effect (and
      (not (agent-at ?from))
      (agent-at ?to)
    )
  )

  ;; =========================================================================
  ;; MANIPULATION
  ;; =========================================================================

  (:action pick-up
    :parameters (?i - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (at ?i ?l)
      (hand-empty)
      (clear ?i)  ; Nothing on top
    )
    :effect (and
      (not (at ?i ?l))
      (not (hand-empty))
      (holding ?i)
    )
  )

  (:action put-down
    :parameters (?i - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (holding ?i)
    )
    :effect (and
      (not (holding ?i))
      (hand-empty)
      (at ?i ?l)
      (clear ?i)  ; When put down, it's clear
    )
  )

  (:action stack
    :parameters (?top - item ?bottom - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (holding ?top)
      (at ?bottom ?l)
      (clear ?bottom)  ; Bottom must be clear
      (is-surface ?bottom)  ; Can only stack on surfaces
      (not (= ?top ?bottom))
    )
    :effect (and
      (not (holding ?top))
      (hand-empty)
      (at ?top ?l)
      (on-top ?top ?bottom)
      (not (clear ?bottom))  ; Bottom is no longer clear
      (clear ?top)  ; Top is clear
    )
  )

  (:action unstack
    :parameters (?top - item ?bottom - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (at ?top ?l)
      (on-top ?top ?bottom)
      (clear ?top)  ; Top must be clear
      (hand-empty)
    )
    :effect (and
      (not (at ?top ?l))
      (not (on-top ?top ?bottom))
      (clear ?bottom)  ; Bottom becomes clear
      (not (hand-empty))
      (holding ?top)
    )
  )

  ;; =========================================================================
  ;; TOOL ACTIONS
  ;; =========================================================================

  (:action slice-object
    :parameters (?obj - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (holding knife)
      (at ?obj ?l)
      (whole ?obj)
      (clear ?obj)  ; Must be clear to slice
      (not (is-tool ?obj))
    )
    :effect (and
      (not (whole ?obj))
      (cut-into-pieces ?obj)
    )
  )

  (:action take-photo
    :parameters (?obj - item ?l - location)
    :precondition (and
      (agent-at ?l)
      (holding dslr)
      (at ?obj ?l)
      (clear ?obj)  ; Must be clear to photograph
      (not (is-tool ?obj))
    )
    :effect (documented ?obj)
  )
)
