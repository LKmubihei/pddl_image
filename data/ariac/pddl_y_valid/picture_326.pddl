(define (problem picture_326)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear green_battery)
        (part_at red_pump table)
        (part_at green_battery table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_battery battery_placement)
        )
    )
)