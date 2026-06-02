(define (problem picture_329)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (part_at green_battery table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
        )
    )
)