(define (problem picture_400)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_battery_1 red_pump green_regulator green_regulator_1 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery_1)
        (clear green_regulator)
        (clear green_regulator_1)
        (clear blue_battery)
        (on green_regulator_1 red_pump)
        (part_at red_pump table)
        (part_at blue_battery_1 table)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)