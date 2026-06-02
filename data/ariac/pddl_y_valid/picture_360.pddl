(define (problem picture_360)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator green_battery red_pump green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_regulator)
        (clear green_battery)
        (clear red_pump)
        (clear green_pump)
        (part_at blue_regulator table)
        (part_at green_battery table)
        (part_at red_pump table)
        (part_at green_pump table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)