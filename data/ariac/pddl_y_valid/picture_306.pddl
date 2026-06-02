(define (problem picture_306)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_regulator blue_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_regulator)
        (clear blue_battery)
        (part_at blue_regulator table)
        (part_at red_pump buffer_placement)
        (part_at blue_battery battery_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)