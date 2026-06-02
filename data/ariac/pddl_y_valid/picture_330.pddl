(define (problem picture_330)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
        )
    )
)