(define (problem picture_183)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_regulator red_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_regulator)
        (clear red_pump)
        (part_at red_pump table)
        (part_at blue_regulator regulator_placement)
        (part_at red_battery battery_placement)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
