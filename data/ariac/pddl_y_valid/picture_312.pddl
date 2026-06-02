(define (problem picture_312)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_regulator green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear green_pump)
        (clear blue_battery)
        (clear red_pump)
        (part_at green_pump table)
        (part_at green_regulator table)
        (part_at blue_battery battery_placement)
        (part_at red_pump buffer_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)