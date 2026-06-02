(define (domain ariac)
  (:requirements :strips :typing :adl)
  (:types
    part location - object
  )

  (:predicates
    (robot_at ?l - location)
    (part_at  ?p - part ?l - location)
    (on ?top - part ?bottom - object)
    (clear ?x - object)
    (holding ?p - part)
    (handempty)
  )

  (:action moveto
    :parameters (?from - location ?to - location)
    :precondition (and
      (robot_at ?from)
    )
    :effect (and
      (robot_at ?to)
      (not (robot_at ?from))
    )
  )

  (:action pick
    :parameters (?p - part ?l - location)
    :precondition (and
      (part_at ?p ?l)
      (robot_at ?l)
      (clear ?p)
      (handempty)
    )
    :effect (and
      (holding ?p)
      (not (handempty))   
    )
  )

  (:action place
    :parameters (?p - part ?target - location)
    :precondition (and
      (holding ?p)
      (robot_at ?target)
    )
    :effect (and
      (part_at ?p ?target)
      (not (holding ?p))
      (handempty)
      (clear ?p)             
    )
  )

  (:action unstack
    :parameters (?top - part ?bottom - part ?l - location)
    :precondition (and
      (on ?top ?bottom)
      (clear ?top)
      (handempty)
      (part_at ?bottom ?l) 
      (robot_at ?l)
    )
    :effect (and
      (holding ?top)
      (clear ?bottom)
      (not (on ?top ?bottom))
      (not (handempty))
    )
  )
)
