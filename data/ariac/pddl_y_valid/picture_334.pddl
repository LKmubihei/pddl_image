(define (problem picture_334)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (clear green_battery)
        (part_at green_pump table)
        (part_at green_battery buffer_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
        )
    )
)